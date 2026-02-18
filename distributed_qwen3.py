import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import time
import os
from collections import deque

from rebalance_algo import DeviceStatus, rebalance_intra_stage

# --- Configuration & Helpers ---

def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1

class DistributedConfig:
    def __init__(self, rank, world_size, stage_ranks, tp_ranks_per_stage):
        """
        stage_ranks: List of lists. e.g. [[0, 1], [2, 3, 4]]
        tp_ranks_per_stage: List of TP sizes. e.g. [2, 3]
        """
        self.rank = rank
        self.world_size = world_size
        self.stage_ranks = stage_ranks
        self.tp_ranks_per_stage = tp_ranks_per_stage
        
        # Determine Stage ID and TP Rank
        self.stage_id = -1
        self.tp_rank = -1
        self.tp_group = None
        self.pp_group = None # Communicator for P2P? Actually P2P uses global rank usually.
        self.is_last_stage = False
        self.is_first_stage = False
        
        for i, ranks in enumerate(stage_ranks):
            if rank in ranks:
                self.stage_id = i
                self.tp_rank = ranks.index(rank)
                self.tp_size = len(ranks)
                self.is_first_stage = (i == 0)
                self.is_last_stage = (i == len(stage_ranks) - 1)
                break
        
        assert self.stage_id != -1, f"Rank {rank} not assigned to any stage"

    def setup_groups(self):
        # Create TP Groups
        # In gloo, new_group is collective. All ranks must call it.
        # But we only need groups for ranks within the same stage.
        # It's safer to create all groups on all ranks to ensure synchronization if needed,
        # but for disjoint groups, we can just create the one we belong to?
        # Standard practice: create all subgroups.
        
        self.tp_groups = []
        for ranks in self.stage_ranks:
            g = dist.new_group(ranks=ranks, backend='gloo')
            if self.rank in ranks:
                self.tp_group = g
        
        # PP Group is usually just global P2P. We don't strictly need a group for send/recv 
        # unless we do specific collective comms across stages.
        # We will use dist.send/recv with global ranks.

# --- Components ---

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype
        if self.qwen3_compatible:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale
        if self.shift is not None:
            norm_x = norm_x + self.shift
        return norm_x.to(input_dtype)

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin

def apply_rope(x, cos, sin, offset=0):
    batch_size, num_heads, seq_len, head_dim = x.shape
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2:]
    cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)

class DistributedGroupedQueryAttention(nn.Module):
    def __init__(self, cfg, dist_config: DistributedConfig):
        super().__init__()
        self.dist_config = dist_config
        self.num_heads = cfg["n_heads"]
        self.num_kv_groups = cfg["n_kv_groups"]
        self.head_dim = cfg["head_dim"] if cfg["head_dim"] is not None else cfg["emb_dim"] // self.num_heads
        self.group_size = self.num_heads // self.num_kv_groups
        self.d_out = self.num_heads * self.head_dim
        self.d_in = cfg["emb_dim"]
        
        # Full Weights (Each device loads full weights for its stage)
        # Enable bias for Q, K, V
        self.W_query = nn.Linear(self.d_in, self.d_out, bias=True, dtype=cfg["dtype"])
        self.W_key = nn.Linear(self.d_in, self.num_kv_groups * self.head_dim, bias=True, dtype=cfg["dtype"])
        self.W_value = nn.Linear(self.d_in, self.num_kv_groups * self.head_dim, bias=True, dtype=cfg["dtype"])
        self.out_proj = nn.Linear(self.d_out, self.d_in, bias=False, dtype=cfg["dtype"])
        
        if cfg["qk_norm"]:
            self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

        # State for Dynamic Load Balancing
        self.active_q_heads_indices = None
        self.maintained_q_heads_indices = None
        self.maintained_kv_indices = None
        
        # Initialize default split (static)
        # Default: Even split if not specified
        self.set_default_split()

    def set_default_split(self):
        # Default: Split Q heads evenly among TP ranks
        q_heads_per_rank = self.num_heads // self.dist_config.tp_size
        start = self.dist_config.tp_rank * q_heads_per_rank
        end = start + q_heads_per_rank
        # Handle remainder for last rank
        if self.dist_config.tp_rank == self.dist_config.tp_size - 1:
            end = self.num_heads
            
        self.maintained_q_heads_indices = list(range(start, end))
        self.active_q_heads_indices = list(range(start, end))
        self.update_kv_indices()

    def set_maintained_heads(self, heads_list):
        """Set the static redundancy superset of Q heads this device maintains KV for."""
        self.maintained_q_heads_indices = sorted(list(heads_list))
        self.update_kv_indices()

    def set_active_heads(self, heads_list):
        """Set the dynamic active subset of Q heads this device computes."""
        # Active heads must be a subset of maintained heads? 
        # Requirement: "active_q_heads... select a subset from maintained_q_heads"
        # So yes, we should validate or just intersect.
        self.active_q_heads_indices = [h for h in heads_list if h in self.maintained_q_heads_indices]
        if len(self.active_q_heads_indices) != len(heads_list):
            # Warn or just proceed? For now, we strictly follow the subset rule.
            pass

    def set_split_by_ratio(self, ratios):
        """
        ratios: List of weights for each rank in this stage. e.g. [3, 3] or [1, 2, 1]
        Sets the ACTIVE heads based on ratio.
        """
        assert len(ratios) == self.dist_config.tp_size
        total_ratio = sum(ratios)
        
        # Calculate Q head ranges for ALL ranks to find my range
        cumulative_heads = 0
        my_start = 0
        my_end = 0
        
        for i, r in enumerate(ratios):
            count = int((r / total_ratio) * self.num_heads)
            if i == len(ratios) - 1:
                count = self.num_heads - cumulative_heads
            
            if i == self.dist_config.tp_rank:
                my_start = cumulative_heads
                my_end = cumulative_heads + count
            
            cumulative_heads += count
            
        new_active = list(range(my_start, my_end))
        self.set_active_heads(new_active)
        
    def update_kv_indices(self):
        # Calculate required KV heads for maintained Q heads
        # Q_head_idx // group_size -> KV_head_idx
        kv_indices = set()
        for q_idx in self.maintained_q_heads_indices:
            kv_idx = q_idx // self.group_size
            kv_indices.add(kv_idx)
        self.maintained_kv_indices = sorted(list(kv_indices))

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        b, num_tokens, _ = x.shape
        
        # 1. Compute Maintained KV (Redundancy Maintenance)
        # We need to compute K, V for all maintained KV heads.
        # We slice the weights manually.
        
        # W_key shape: (d_in, num_kv_groups * head_dim) -> We need to select columns.
        # But nn.Linear stores weight as (out_features, in_features).
        # So W_key.weight shape is (num_kv_groups * head_dim, d_in).
        # We need to select rows corresponding to maintained_kv_indices.
        
        kv_head_indices = torch.tensor(self.maintained_kv_indices, device=x.device)
        # Each KV head has `head_dim` rows.
        # We need to construct the row indices.
        # shape of indices: (num_maintained_kv * head_dim)
        
        # Optimization: pre-compute indices tensor
        kv_weight_indices = []
        for i in self.maintained_kv_indices:
            kv_weight_indices.extend(range(i * self.head_dim, (i + 1) * self.head_dim))
        kv_weight_indices = torch.tensor(kv_weight_indices, device=x.device)
        
        # Slice Weights
        # Note: In a real optimized system, we wouldn't index the weight matrix every step if it's static.
        # But here we simulate the logic.
        k_weight_subset = self.W_key.weight[kv_weight_indices] # (subset_dim, d_in)
        k_bias_subset = self.W_key.bias[kv_weight_indices] if self.W_key.bias is not None else None
        
        v_weight_subset = self.W_value.weight[kv_weight_indices]
        v_bias_subset = self.W_value.bias[kv_weight_indices] if self.W_value.bias is not None else None
        
        keys = F.linear(x, k_weight_subset, k_bias_subset) # (b, seq, subset_dim)
        values = F.linear(x, v_weight_subset, v_bias_subset)
        
        # 2. Compute Active Q
        q_weight_indices = []
        for i in self.active_q_heads_indices:
            q_weight_indices.extend(range(i * self.head_dim, (i + 1) * self.head_dim))
        q_weight_indices = torch.tensor(q_weight_indices, device=x.device)
        
        q_weight_subset = self.W_query.weight[q_weight_indices]
        q_bias_subset = self.W_query.bias[q_weight_indices] if self.W_query.bias is not None else None
        
        queries = F.linear(x, q_weight_subset, q_bias_subset)
        
        # 3. Reshape and RoPE
        # Queries: (b, seq, num_active_q, head_dim)
        queries = queries.view(b, num_tokens, len(self.active_q_heads_indices), self.head_dim).transpose(1, 2)
        # Keys/Values: (b, seq, num_maintained_kv, head_dim)
        keys_new = keys.view(b, num_tokens, len(self.maintained_kv_indices), self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, len(self.maintained_kv_indices), self.head_dim).transpose(1, 2)
        
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)
            
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)
        
        # 4. KV Cache Update
        # We only update the cache for maintained KV heads.
        # The cache structure needs to handle partial updates or we store sparse?
        # Requirement: "Redundancy Maintenance". 
        # Ideally, `cache` object should store the full KV state, but distributed?
        # Or does each rank store only its maintained part?
        # "Device responsible for maintaining Cache for these Q-Heads superset"
        # So each device has a local cache for its maintained KV heads.
        
        if cache is not None:
            # cache is (k, v) for this layer
            # In this distributed setup, `cache` passed in should probably be a local structure
            # containing only the maintained heads.
            # If it's the first step, it might be empty.
            if len(cache) == 0:
                prev_k = torch.empty(b, len(self.maintained_kv_indices), 0, self.head_dim, device=x.device, dtype=keys_new.dtype)
                prev_v = torch.empty(b, len(self.maintained_kv_indices), 0, self.head_dim, device=x.device, dtype=values_new.dtype)
            else:
                prev_k, prev_v = cache
            
            # Append new
            keys_cached = torch.cat([prev_k, keys_new], dim=2)
            values_cached = torch.cat([prev_v, values_new], dim=2)
            next_cache = (keys_cached, values_cached)
        else:
            keys_cached, values_cached = keys_new, values_new
            next_cache = (keys_cached, values_cached)
            
        # 5. Attention
        # We need to map Active Q to the Cached KV.
        # `active_q_heads_indices` are global indices.
        # `maintained_kv_indices` are global indices.
        # We need to find which index in `keys_cached` corresponds to the KV head for each Q head.
        
        # Construct a gather map
        # For each q in active_q, find kv index.
        # Then find position of that kv index in maintained_kv_indices.
        
        kv_map_indices = []
        for q_idx in self.active_q_heads_indices:
            kv_global_idx = q_idx // self.group_size
            try:
                local_kv_pos = self.maintained_kv_indices.index(kv_global_idx)
                kv_map_indices.append(local_kv_pos)
            except ValueError:
                raise RuntimeError(f"Q head {q_idx} needs KV head {kv_global_idx} which is not in maintained list {self.maintained_kv_indices}")
        
        kv_map_indices = torch.tensor(kv_map_indices, device=x.device)
        
        # Select specific KV for each Q (Repeat/Interleave logic)
        # keys_cached: (b, n_m_kv, seq_total, head_dim)
        # we want keys_for_q: (b, n_active_q, seq_total, head_dim)
        
        keys_for_attn = keys_cached.index_select(1, kv_map_indices)
        values_for_attn = values_cached.index_select(1, kv_map_indices)
        
        # Attention
        attn_scores = queries @ keys_for_attn.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        
        context = (attn_weights @ values_for_attn).transpose(1, 2).reshape(b, num_tokens, len(self.active_q_heads_indices) * self.head_dim)
        
        # 6. Output Projection (Row Parallel-ish)
        # We computed a partial result. The output dimension is `active_heads * head_dim`.
        # We need to project this back to `d_in`.
        # `out_proj` has shape (d_in, d_out).
        # We need to select columns of `out_proj` corresponding to `active_q_heads`.
        
        out_proj_indices = []
        for i in self.active_q_heads_indices:
            out_proj_indices.extend(range(i * self.head_dim, (i + 1) * self.head_dim))
        out_proj_indices = torch.tensor(out_proj_indices, device=x.device)
        
        # Weight shape (d_in, d_out). We select columns (dim 1).
        out_weight_subset = self.out_proj.weight[:, out_proj_indices] # (d_in, subset_dim)
        
        # context shape (b, seq, subset_dim)
        # output = context @ out_weight_subset.T
        output = F.linear(context, out_weight_subset) # (b, seq, d_in)
        
        # 7. AllReduce happens outside in the Block?
        # Requirement: "Attention output经过AllReduce汇总后..."
        # Yes, return partial output.
        
        return output, next_cache

class DistributedFeedForward(nn.Module):
    def __init__(self, cfg, dist_config: DistributedConfig):
        super().__init__()
        self.dist_config = dist_config
        self.emb_dim = cfg["emb_dim"]
        self.hidden_dim = cfg["hidden_dim"]
        self.dtype = cfg["dtype"]
        
        # Full weights
        self.fc1 = nn.Linear(self.emb_dim, self.hidden_dim, dtype=self.dtype, bias=False)
        self.fc2 = nn.Linear(self.emb_dim, self.hidden_dim, dtype=self.dtype, bias=False)
        self.fc3 = nn.Linear(self.hidden_dim, self.emb_dim, dtype=self.dtype, bias=False)
        
        # Default split: even split of hidden_dim
        self.set_default_split()
        
    def set_default_split(self):
        # Even split of hidden_dim
        per_rank = self.hidden_dim // self.dist_config.tp_size
        start = self.dist_config.tp_rank * per_rank
        end = start + per_rank
        if self.dist_config.tp_rank == self.dist_config.tp_size - 1:
            end = self.hidden_dim
        self.active_hidden_indices = list(range(start, end))
        
    def set_active_range(self, start, end):
        """Set the dynamic active range of hidden dims this device computes."""
        self.active_hidden_indices = list(range(start, end))

    def set_split_by_ratio(self, ratios):
        total_ratio = sum(ratios)
        cumulative = 0
        my_start = 0
        my_end = 0
        for i, r in enumerate(ratios):
            count = int((r / total_ratio) * self.hidden_dim)
            if i == len(ratios) - 1:
                count = self.hidden_dim - cumulative
            if i == self.dist_config.tp_rank:
                my_start = cumulative
                my_end = cumulative + count
            cumulative += count
        self.active_hidden_indices = list(range(my_start, my_end))

    def forward(self, x):
        # 1. Column Parallel (fc1, fc2)
        # Select rows of weight matrix (out_features, in_features)
        indices = torch.tensor(self.active_hidden_indices, device=x.device)
        
        w1_subset = self.fc1.weight[indices]
        w2_subset = self.fc2.weight[indices]
        
        x_fc1 = F.linear(x, w1_subset)
        x_fc2 = F.linear(x, w2_subset)
        x_hidden = F.silu(x_fc1) * x_fc2
        
        # 2. Row Parallel (fc3)
        # Select columns of weight matrix (out_features, in_features)
        # fc3 weight is (emb_dim, hidden_dim). Select columns matching indices.
        w3_subset = self.fc3.weight[:, indices]
        
        out = F.linear(x_hidden, w3_subset)
        
        return out

class DistributedTransformerBlock(nn.Module):
    def __init__(self, cfg, dist_config):
        super().__init__()
        self.dist_config = dist_config
        self.att = DistributedGroupedQueryAttention(cfg, dist_config)
        self.ff = DistributedFeedForward(cfg, dist_config)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.tp_group = dist_config.tp_group

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None, skip_final_allreduce=False):
        # Attention Block
        shortcut = x
        x_norm = self.norm1(x)
        
        # Start Timing
        t0 = time.time()
        
        # Partial Output
        x_partial, next_cache = self.att(x_norm, mask, cos, sin, start_pos=start_pos, cache=cache)
        
        # AllReduce
        if self.dist_config.tp_size > 1:
            dist.all_reduce(x_partial, op=dist.ReduceOp.SUM, group=self.tp_group)
            
        x = x_partial + shortcut
        
        # FFN Block
        shortcut = x
        x_norm = self.norm2(x)
        
        x_partial = self.ff(x_norm)
        
        # AllReduce (Conditional for last layer)
        if self.dist_config.tp_size > 1:
            if not skip_final_allreduce:
                dist.all_reduce(x_partial, op=dist.ReduceOp.SUM, group=self.tp_group)
                x = x_partial + shortcut
                return x, None, next_cache
            else:
                # Return partial FFN output and shortcut separately
                # The caller (Model) will handle the Reduce-to-Root and addition
                return x_partial, shortcut, next_cache
        else:
            # TP=1
            x = x_partial + shortcut
            return x, None, next_cache

class DistributedQwen3Model(nn.Module):
    def __init__(self, cfg, dist_config: DistributedConfig, managed_layers_range=None):
        super().__init__()
        self.cfg = cfg
        self.dist_config = dist_config
        
        # Determine layers for this stage
        total_layers = cfg["n_layers"]
        
        if managed_layers_range is not None:
            # Manual override for managed layers (Loaded)
            start_layer, end_layer = managed_layers_range
            self.layer_start = start_layer
            self.layer_end = end_layer
            my_num_layers = end_layer - start_layer
            self.my_layers = range(start_layer, end_layer)
        else:
            # Simple static layer partitioning for Pipeline (Default)
            num_stages = len(dist_config.stage_ranks)
            layers_per_stage = total_layers // num_stages
            remainder = total_layers % num_stages
            
            # Calculate my layer range
            start_layer = 0
            for i in range(dist_config.stage_id):
                start_layer += layers_per_stage + (1 if i < remainder else 0)
            
            my_num_layers = layers_per_stage + (1 if dist_config.stage_id < remainder else 0)
            end_layer = start_layer + my_num_layers
            
            self.layer_start = start_layer
            self.layer_end = end_layer
            self.my_layers = range(start_layer, end_layer)
        
        # Active layers (Dynamic Execution) - initially same as loaded
        self.active_layer_start = self.layer_start
        self.active_layer_end = self.layer_end
        
        print(f"Rank {dist_config.rank} (Stage {dist_config.stage_id}) loading layers {self.layer_start}-{self.layer_end-1}")

        # Components
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.ModuleList([
            DistributedTransformerBlock(cfg, dist_config) for _ in range(my_num_layers)
        ])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # RoPE
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
        self.current_pos = 0
        
        # --- Dynamic Load Balancing State ---
        self.tp_size = dist_config.tp_size
        self.tp_rank = dist_config.tp_rank
        
        # Initial Counts (Even Split)
        self.head_counts = [cfg["n_heads"] // self.tp_size] * self.tp_size
        for i in range(cfg["n_heads"] % self.tp_size):
            self.head_counts[i] += 1
            
        self.ffn_counts = [cfg["hidden_dim"] // self.tp_size] * self.tp_size
        for i in range(cfg["hidden_dim"] % self.tp_size):
            self.ffn_counts[i] += 1
            
        # KV Holding is static and same as initial head split
        self.kv_head_counts = list(self.head_counts) 
        
        # Helper to compute ranges
        def counts_to_ranges(counts):
            ranges = []
            start = 0
            for c in counts:
                ranges.append((start, start + c))
                start += c
            return ranges
            
        self.kv_ranges = counts_to_ranges(self.kv_head_counts)

    def load_weights_from_hf(self, model_path):
        """
        Load weights from a Hugging Face model directory (safetensors).
        Only loads weights relevant to this stage/rank.
        """
        import json
        from safetensors import safe_open
        
        # 1. Load Config (optional verification)
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                hf_config = json.load(f)
            
        # 2. Iterate over safetensors files
        # Check model.safetensors.index.json if it exists (sharded)
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        weight_files = []
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)
            weight_files = sorted(list(set(index["weight_map"].values())))
        else:
            # Assume single file or check for model.safetensors
            if os.path.exists(os.path.join(model_path, "model.safetensors")):
                weight_files = ["model.safetensors"]
            else:
                # Try to find any .safetensors
                files = os.listdir(model_path)
                weight_files = [f for f in files if f.endswith(".safetensors")]
        
        print(f"Rank {self.dist_config.rank}: Loading weights from {weight_files}")
        
        for w_file in weight_files:
            file_path = os.path.join(model_path, w_file)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    self._load_single_weight(key, f.get_tensor(key))
                    
    def _load_single_weight(self, key, tensor):
        # Map HF key to DistributedQwen3Model key
        # HF: model.layers.0.self_attn.q_proj.weight
        # My: trf_blocks.0.att.W_query.weight
        
        # Helper to assign
        def assign(param, data):
            if param.shape != data.shape:
                # Transpose check for Linear weights? 
                # HF Linear weights are (out, in). nn.Linear weights are (out, in).
                # But some models might be transposed? Qwen is standard.
                pass
            with torch.no_grad():
                param.copy_(data)

        if "model.layers" in key:
            parts = key.split(".")
            layer_idx = int(parts[2])
            
            # Check if this layer belongs to this stage
            if self.layer_start <= layer_idx < self.layer_end:
                local_idx = layer_idx - self.layer_start
                block = self.trf_blocks[local_idx]
                module_name = parts[3] # self_attn or mlp
                
                # Determine if weight or bias
                # parts: model.layers.0.self_attn.q_proj.weight -> last is weight
                param_type = parts[-1] # "weight" or "bias"
                
                if module_name == "self_attn":
                    proj = parts[4] # q_proj, k_proj, v_proj, o_proj
                    
                    target_module = None
                    if proj == "q_proj": target_module = block.att.W_query
                    elif proj == "k_proj": target_module = block.att.W_key
                    elif proj == "v_proj": target_module = block.att.W_value
                    elif proj == "o_proj": target_module = block.att.out_proj
                    
                    if target_module is not None:
                        if param_type == "bias" and target_module.bias is not None:
                            assign(target_module.bias, tensor)
                        elif param_type == "weight":
                            assign(target_module.weight, tensor)
                            
                elif module_name == "mlp":
                    proj = parts[4] # gate_proj, up_proj, down_proj
                    
                    target_module = None
                    if proj == "gate_proj": target_module = block.ff.fc1
                    elif proj == "up_proj": target_module = block.ff.fc2
                    elif proj == "down_proj": target_module = block.ff.fc3
                    
                    if target_module is not None:
                         if param_type == "bias" and target_module.bias is not None:
                            assign(target_module.bias, tensor)
                         elif param_type == "weight":
                            assign(target_module.weight, tensor)

                elif module_name == "input_layernorm":
                    if param_type == "weight": # HF uses 'weight' for RMSNorm scale usually? Or 'gamma'? Safetensors usually 'weight'.
                        # Wait, code used `assign(block.norm1.scale, tensor)`
                        # RMSNorm has `scale`.
                        assign(block.norm1.scale, tensor)
                elif module_name == "post_attention_layernorm":
                    if param_type == "weight":
                        assign(block.norm2.scale, tensor)
                    
        elif "model.embed_tokens" in key:
            if self.dist_config.is_first_stage:
                assign(self.tok_emb.weight, tensor)
            
            # Handle tied weights for output head (if lm_head is missing, this serves as fallback)
            # If lm_head exists, it will be loaded later and overwrite this (assuming alphabetical order or explicit check)
            # To be safe, we can check if lm_head key exists in the file, but across files is hard.
            # We'll just load it. If lm_head comes later, it overwrites.
            if self.dist_config.is_last_stage:
                assign(self.out_head.weight, tensor)
                
        elif "model.norm" in key:
            if self.dist_config.is_last_stage:
                assign(self.final_norm.scale, tensor)
                
        elif "lm_head" in key:
            if self.dist_config.is_last_stage:
                assign(self.out_head.weight, tensor)

    def apply_load_balance_policy(self):
        """Apply current head_counts and ffn_counts to all blocks."""
        # Calculate my range based on tp_rank and counts
        my_head_start = sum(self.head_counts[:self.tp_rank])
        my_head_end = my_head_start + self.head_counts[self.tp_rank]
        
        my_ffn_start = sum(self.ffn_counts[:self.tp_rank])
        my_ffn_end = my_ffn_start + self.ffn_counts[self.tp_rank]
        
        active_heads = list(range(my_head_start, my_head_end))
        
        for block in self.trf_blocks:
            # Apply to Attention
            block.att.set_active_heads(active_heads)
            # Apply to FFN
            block.ff.set_active_range(my_ffn_start, my_ffn_end)

    def forward(self, x_or_idx, cache=None, runtime_layer_counts=None, start_pos=None, force_rebalance=False):
        """
        runtime_layer_counts: List of integers, e.g. [10, 12, 14], indicating layers per stage.
        If provided, updates the active layer range for this forward pass.
        start_pos: Current position in sequence (for RoPE/Mask). If None, inferred from cache.
        force_rebalance: If True, forces a rebalance check (simulated imbalance).
        """
        # 0. Apply Dynamic Load Balancing Strategy
        self.apply_load_balance_policy()
        
        # 1. Update Dynamic Execution Logic (Layer Range)
        if runtime_layer_counts is not None:
            # Validate input
            assert len(runtime_layer_counts) == len(self.dist_config.stage_ranks), \
                f"Layer counts {len(runtime_layer_counts)} must match stage count {len(self.dist_config.stage_ranks)}"
            
            # Calculate my active range
            active_start = 0
            for i in range(self.dist_config.stage_id):
                active_start += runtime_layer_counts[i]
            
            active_end = active_start + runtime_layer_counts[self.dist_config.stage_id]
            
            self.active_layer_start = active_start
            self.active_layer_end = active_end
            
        # P2P Receive Input (if not first logical stage)
        if self.dist_config.is_first_stage:
            if self.dist_config.tp_rank == 0:
                x = self.tok_emb(x_or_idx)
            else:
                x = self.tok_emb(x_or_idx)
        else:
            # Receive from previous stage
            if self.dist_config.tp_rank == 0:
                # Receive Metadata (Shape)
                shape_tensor = torch.zeros(3, dtype=torch.long, device='cpu')
                src_rank = self.dist_config.stage_ranks[self.dist_config.stage_id - 1][0]
                dist.recv(shape_tensor, src=src_rank)
                
                # Allocate Buffer
                x = torch.empty(tuple(shape_tensor.tolist()), dtype=self.cfg["dtype"]) # CPU receive
                dist.recv(x, src=src_rank)
                x = x.to(next(self.parameters()).device)
            else:
                x = None
            
            # Broadcast to TP group
            if self.dist_config.tp_size > 1:
                # Broadcast Shape
                if self.dist_config.tp_rank == 0:
                    shape_tensor = torch.tensor(x.shape, device=next(self.parameters()).device)
                else:
                    shape_tensor = torch.zeros(3, dtype=torch.long, device=next(self.parameters()).device)
                
                dist.broadcast(shape_tensor, src=self.dist_config.stage_ranks[self.dist_config.stage_id][0], group=self.dist_config.tp_group)
                
                if self.dist_config.tp_rank != 0:
                    x = torch.empty(tuple(shape_tensor.tolist()), dtype=self.cfg["dtype"], device=next(self.parameters()).device)
                
                dist.broadcast(x, src=self.dist_config.stage_ranks[self.dist_config.stage_id][0], group=self.dist_config.tp_group)

        # 2. Forward Pass
        # Ensure x is on device
        if x.device.type == 'cpu':
             x = x.to(next(self.parameters()).device)

        num_tokens = x.shape[1]
        
        # Mask setup
        # Infer start_pos if not provided
        if start_pos is None:
            start_pos = 0
            if cache:
                # Iterate over all loaded layers to find max length
                max_len = 0
                for i in range(len(self.trf_blocks)):
                    c = cache.get(i)
                    if c and len(c) > 0 and c[0].numel() > 0:
                        max_len = max(max_len, c[0].shape[2])
                start_pos = max_len
        
        mask = torch.triu(
            torch.ones(num_tokens + start_pos, num_tokens + start_pos, device=x.device, dtype=torch.bool), diagonal=1
        )[start_pos : start_pos + num_tokens, : start_pos + num_tokens]
        mask = mask[None, None, :, :]

        # Start Timing
        t_start = time.time()

        # Iterate over ACTIVE layers only
        for global_layer_idx in range(self.active_layer_start, self.active_layer_end):
            # Map global to local
            if global_layer_idx < self.layer_start or global_layer_idx >= self.layer_end:
                # Skip if not loaded
                continue
                
            local_idx = global_layer_idx - self.layer_start
            block = self.trf_blocks[local_idx]
            
            blk_cache = cache.get(local_idx) if cache else None
            
            # Handle cold start or partial history
            current_cache_len = 0
            if blk_cache and len(blk_cache) > 0 and blk_cache[0].numel() > 0:
                current_cache_len = blk_cache[0].shape[2]
            
            if start_pos > current_cache_len:
                 layer_start_pos = current_cache_len
                 layer_mask = torch.triu(
                    torch.ones(num_tokens + layer_start_pos, num_tokens + layer_start_pos, device=x.device, dtype=torch.bool), diagonal=1
                 )[layer_start_pos : layer_start_pos + num_tokens, : layer_start_pos + num_tokens]
                 layer_mask = layer_mask[None, None, :, :]
            else:
                 layer_start_pos = start_pos
                 layer_mask = mask
            
            # Optimization: Skip final AllReduce for the last active layer
            is_last_active = (global_layer_idx == self.active_layer_end - 1)
            skip_final = is_last_active and self.tp_size > 1
            
            x, shortcut, new_blk_cache = block(x, layer_mask, self.cos, self.sin, start_pos=layer_start_pos, cache=blk_cache, skip_final_allreduce=skip_final)
            
            if cache:
                cache.update(local_idx, new_blk_cache)
                
            if skip_final:
                # Reduce to Root (x is partial FFN)
                dist.reduce(x, dst=self.dist_config.stage_ranks[self.dist_config.stage_id][0], op=dist.ReduceOp.SUM, group=self.dist_config.tp_group)
                if self.dist_config.tp_rank == 0:
                    x = x + shortcut
                else:
                    x = None

        t_end = time.time()
        exec_time = (t_end - t_start) * 1000.0 # ms
        
        if force_rebalance:
            # Fake imbalance
            if self.tp_rank == 0: exec_time = 100.0
            else: exec_time = 50.0

        # 3. Dynamic Load Balancing (Post-Run)
        # Root gathers times and updates policy
        if self.tp_size > 1:
            times = [torch.zeros(1, device=x.device if x is not None else 'cpu') for _ in range(self.tp_size)]
            my_time = torch.tensor([exec_time], device=x.device if x is not None else 'cpu')
            
            dist.all_gather(times, my_time, group=self.dist_config.tp_group)
            times_ms = [t.item() for t in times]
            
            if self.tp_rank == 0:
                # Root runs algorithm
                devices = []
                # Reconstruct current ranges for the algorithm input
                # Use self.head_counts/ffn_counts directly as they track current state
                curr_h = 0
                curr_f = 0
                for r in range(self.tp_size):
                    # Heads
                    h_count = self.head_counts[r]
                    h_range = (curr_h, curr_h + h_count)
                    curr_h += h_count
                    
                    # FFN
                    f_count = self.ffn_counts[r]
                    f_range = (curr_f, curr_f + f_count)
                    curr_f += f_count
                    
                    d = DeviceStatus(
                        device_id=r,
                        execution_time_ms=times_ms[r],
                        current_head_start=h_range[0],
                        current_head_end=h_range[1],
                        current_ffn_start=f_range[0],
                        current_ffn_end=f_range[1],
                        kv_head_holding_start=self.kv_ranges[r][0],
                        kv_head_holding_end=self.kv_ranges[r][1]
                    )
                    devices.append(d)
                
                new_head_counts, new_ffn_counts = rebalance_intra_stage(devices)
                
                # Check for change
                if new_head_counts != self.head_counts or new_ffn_counts != self.ffn_counts:
                    print(f"Stage {self.dist_config.stage_id} Rebalance Triggered!")
                    print(f"  Old Heads: {self.head_counts}, New Heads: {new_head_counts}")
                    print(f"  Old FFN: {self.ffn_counts}, New FFN: {new_ffn_counts}")
                
                if not new_head_counts: # Safety
                    new_head_counts = self.head_counts
                    new_ffn_counts = self.ffn_counts
                    
                self.head_counts = new_head_counts
                self.ffn_counts = new_ffn_counts
                
                # Broadcast
                counts_tensor = torch.tensor(self.head_counts + self.ffn_counts, device=x.device if x is not None else 'cpu', dtype=torch.long)
            else:
                counts_tensor = torch.zeros(self.tp_size * 2, device=x.device if x is not None else 'cpu', dtype=torch.long)
            
            dist.broadcast(counts_tensor, src=self.dist_config.stage_ranks[self.dist_config.stage_id][0], group=self.dist_config.tp_group)
            
            counts_list = counts_tensor.tolist()
            self.head_counts = counts_list[:self.tp_size]
            self.ffn_counts = counts_list[self.tp_size:]

        # 4. Output / Send
        if self.dist_config.is_last_stage:
            if self.dist_config.tp_rank == 0:
                # Root computes output
                # x might be on CPU if previous step failed? No, x is None on non-root.
                x = self.final_norm(x)
                logits = self.out_head(x)
                return logits
            else:
                return None
        else:
            # Send to next stage
            if self.dist_config.tp_rank == 0:
                dst_rank = self.dist_config.stage_ranks[self.dist_config.stage_id + 1][0]
                # Send shape
                shape_tensor = torch.tensor(x.shape, dtype=torch.long, device='cpu')
                dist.send(shape_tensor, dst=dst_rank)
                # Send data
                dist.send(x.cpu(), dst=dst_rank)
            return None

class LocalKVCache:
    def __init__(self, n_local_layers):
        self.cache = [[] for _ in range(n_local_layers)] # Use empty list to denote uninit? Or None.
    
    def get(self, layer_idx):
        if len(self.cache[layer_idx]) == 0:
            return []
        return self.cache[layer_idx]
    
    def update(self, layer_idx, value):
        self.cache[layer_idx] = value

