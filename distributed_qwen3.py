import os
import json
import argparse
import pickle
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file
from transformers import AutoTokenizer
from init_algorithm import (
    Device, Link, RRAGCConfig, LayerTask, ModelConfig, 
    run_initialization, InitResult
)

# --- Model Components (from Standalone Qwen3) ---

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

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, dtype=cfg.dtype, bias=False)
        self.fc2 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, dtype=cfg.dtype, bias=False)
        self.fc3 = nn.Linear(cfg.hidden_dim, cfg.emb_dim, dtype=cfg.dtype, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)

class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.n_heads
        self.num_kv_groups = cfg.n_kv_groups
        self.head_dim = cfg.head_dim
        self.group_size = self.num_heads // self.num_kv_groups
        self.d_out = self.num_heads * self.head_dim
        
        self.W_query = nn.Linear(cfg.emb_dim, self.d_out, bias=True, dtype=cfg.dtype)
        self.W_key = nn.Linear(cfg.emb_dim, self.num_kv_groups * self.head_dim, bias=True, dtype=cfg.dtype)
        self.W_value = nn.Linear(cfg.emb_dim, self.num_kv_groups * self.head_dim, bias=True, dtype=cfg.dtype)
        self.out_proj = nn.Linear(self.d_out, cfg.emb_dim, bias=False, dtype=cfg.dtype)
        
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6) if cfg.qk_norm else None

    def forward(self, x, mask, cos, sin, start_pos=0):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)
        
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys = apply_rope(keys, cos, sin, offset=start_pos)
        
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)
        
        attn_scores = queries @ keys.transpose(2, 3)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg.emb_dim, eps=1e-6)
        self.norm2 = RMSNorm(cfg.emb_dim, eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin, start_pos) + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x) + shortcut
        return x

# --- Distributed Logic ---

class DistributedConfig:
    def __init__(self, vocab_size=151936, emb_dim=4096, n_layers=32, n_heads=32, n_kv_groups=32, head_dim=128, hidden_dim=11008, rope_base=1000000, context_length=32768, dtype=torch.bfloat16, qk_norm=False):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.rope_base = rope_base
        self.context_length = context_length
        self.dtype = dtype
        self.qk_norm = qk_norm

class DistributedQwen3Model(nn.Module):
    def __init__(self, config, stage_ranks, layers_per_stage, rank, world_size):
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.stage_ranks = stage_ranks
        self.layers_per_stage = layers_per_stage
        
        # Determine my stage
        self.my_stage_idx = -1
        self.my_stage_rank_idx = -1
        for i, ranks in enumerate(stage_ranks):
            if rank in ranks:
                self.my_stage_idx = i
                self.my_stage_rank_idx = ranks.index(rank)
                break
        
        if self.my_stage_idx == -1:
            raise ValueError(f"Rank {rank} not assigned to any stage in {stage_ranks}")
            
        # Determine layers for this stage
        self.start_layer = sum(layers_per_stage[:self.my_stage_idx])
        self.num_layers = layers_per_stage[self.my_stage_idx]
        self.end_layer = self.start_layer + self.num_layers
        
        print(f"[Rank {rank}] Stage {self.my_stage_idx}, Layers {self.start_layer}-{self.end_layer-1} (Total: {self.num_layers})")
        
        # Initialize Local Layers
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(self.num_layers)])
        
        # RoPE
        cos, sin = compute_rope_params(config.head_dim, theta_base=config.rope_base, context_length=config.context_length, dtype=torch.float32)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
        # Embeddings and Output Head (Only on first and last stage respectively? Or simplified)
        # For simplicity, we assume input is already embeddings or we have embeddings on first stage
        if self.my_stage_idx == 0:
            self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim, dtype=config.dtype)
        
        if self.my_stage_idx == len(stage_ranks) - 1:
            self.final_norm = RMSNorm(config.emb_dim)
            self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False, dtype=config.dtype)

    def load_weights(self, model_path):
        print(f"[Rank {self.rank}] Loading weights from {model_path}...")
        state_dict = load_file(model_path)
        
        # Helper to load a specific layer
        def load_layer_weights(layer_idx, block):
            prefix = f"model.layers.{layer_idx}."
            
            # Attention
            block.att.W_query.weight.data.copy_(state_dict[f"{prefix}self_attn.q_proj.weight"])
            block.att.W_query.bias.data.copy_(state_dict[f"{prefix}self_attn.q_proj.bias"])
            
            block.att.W_key.weight.data.copy_(state_dict[f"{prefix}self_attn.k_proj.weight"])
            block.att.W_key.bias.data.copy_(state_dict[f"{prefix}self_attn.k_proj.bias"])
            
            block.att.W_value.weight.data.copy_(state_dict[f"{prefix}self_attn.v_proj.weight"])
            block.att.W_value.bias.data.copy_(state_dict[f"{prefix}self_attn.v_proj.bias"])
            
            block.att.out_proj.weight.data.copy_(state_dict[f"{prefix}self_attn.o_proj.weight"])
            
            if block.att.q_norm:
                block.att.q_norm.scale.data.copy_(state_dict[f"{prefix}self_attn.q_norm.weight"])
            if block.att.k_norm:
                block.att.k_norm.scale.data.copy_(state_dict[f"{prefix}self_attn.k_norm.weight"])

            # FeedForward
            block.ff.fc1.weight.data.copy_(state_dict[f"{prefix}mlp.gate_proj.weight"])
            block.ff.fc2.weight.data.copy_(state_dict[f"{prefix}mlp.up_proj.weight"])
            block.ff.fc3.weight.data.copy_(state_dict[f"{prefix}mlp.down_proj.weight"])
            
            # Norms
            block.norm1.scale.data.copy_(state_dict[f"{prefix}input_layernorm.weight"])
            block.norm2.scale.data.copy_(state_dict[f"{prefix}post_attention_layernorm.weight"])

        # Load Embeddings (Stage 0)
        if self.my_stage_idx == 0:
            self.tok_emb.weight.data.copy_(state_dict["model.embed_tokens.weight"])
            
        # Load Layers
        for i, layer in enumerate(self.layers):
            global_layer_idx = self.start_layer + i
            load_layer_weights(global_layer_idx, layer)
            
        # Load Final Norm & Head (Last Stage)
        if self.my_stage_idx == len(self.stage_ranks) - 1:
            self.final_norm.scale.data.copy_(state_dict["model.norm.weight"])
            
            if "lm_head.weight" in state_dict:
                self.out_head.weight.data.copy_(state_dict["lm_head.weight"])
            elif "model.embed_tokens.weight" in state_dict:
                print(f"[Rank {self.rank}] lm_head.weight not found, using model.embed_tokens.weight (tied weights)")
                self.out_head.weight.data.copy_(state_dict["model.embed_tokens.weight"])
            else:
                raise KeyError("Neither lm_head.weight nor model.embed_tokens.weight found")
            
        print(f"[Rank {self.rank}] Weights loaded successfully.")

    def forward(self, x, start_pos=0):
        # x is input_ids on Stage 0, or hidden_states on other stages
        
        # 1. Receive Input (if not first stage)
        if self.my_stage_idx > 0:
            # Receive from previous stage leader
            prev_stage_leader = self.stage_ranks[self.my_stage_idx - 1][0]
            # Metadata first (shape)
            shape_tensor = torch.zeros(3, dtype=torch.long) # b, seq, dim
            dist.recv(shape_tensor, src=prev_stage_leader)
            b, seq, dim = shape_tensor.tolist()
            
            x = torch.zeros((b, seq, dim), dtype=self.config.dtype)
            dist.recv(x, src=prev_stage_leader)
            
            # Receive start_pos
            start_pos_tensor = torch.zeros(1, dtype=torch.long)
            dist.recv(start_pos_tensor, src=prev_stage_leader)
            start_pos = start_pos_tensor.item()
            
        # x should be valid now
        input_dtype = x.dtype
        
        # 2. Local Processing
        if self.my_stage_idx == 0:
            x = self.tok_emb(x) # Convert ids to embeddings
            
        b, seq, _ = x.shape
        mask = None # Simplified: No mask for now or causal mask
        if seq > 1:
             # Create causal mask for prefill
             mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1).to(x.device)
        
        for layer in self.layers:
            x = layer(x, mask, self.cos, self.sin, start_pos=start_pos)
            
        # 3. Send Output (if not last stage)
        if self.my_stage_idx < len(self.stage_ranks) - 1:
            next_stage_leader = self.stage_ranks[self.my_stage_idx + 1][0]
            
            # Send shape
            shape_tensor = torch.tensor(x.shape, dtype=torch.long)
            dist.send(shape_tensor, dst=next_stage_leader)
            
            # Send data
            dist.send(x, dst=next_stage_leader)
            
            # Send start_pos
            start_pos_tensor = torch.tensor([start_pos], dtype=torch.long)
            dist.send(start_pos_tensor, dst=next_stage_leader)
            
            return None # Return None as work is handed off
            
        # 4. Final Processing (Last Stage)
        else:
            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits

# --- Main Execution ---

def load_cluster_config(config_path):
    with open(config_path, 'r') as f:
        data = json.load(f)
        
    devices = [Device(**d) for d in data["devices"]]
    links = [Link(src_id=l["src"], dst_id=l["dst"], bandwidth=l["bandwidth"]) for l in data["links"]]
    rragc_config = RRAGCConfig(**data["rragc"])
    model_config = ModelConfig(**data["model"])
    layer_task = LayerTask(**data["layer_task"])
    
    return devices, links, rragc_config, layer_task, model_config

def run_distributed_inference():
    parser = argparse.ArgumentParser(description="Distributed Qwen3 Inference")
    parser.add_argument("--mode", type=str, choices=["auto", "manual"], default="auto", help="Initialization mode")
    parser.add_argument("--cluster_config", type=str, default="cluster_config.json", help="Path to cluster config (for auto mode)")
    parser.add_argument("--manual_config", type=str, help="JSON string of stage_ranks (e.g. '[[0,1],[2,3]]') (for manual mode)")
    parser.add_argument("--backend", type=str, default="gloo", help="Distributed backend")
    
    args = parser.parse_args()
    
    # 1. Initialize Distributed Environment
    # Ensure MASTER_ADDR and MASTER_PORT are set (usually by torchrun)
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
        
    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 2. Configuration Phase (Master-Worker)
    config_data = None
    
    if rank == 0:
        if args.mode == "auto":
            print(f"[Rank 0] Running Automatic Initialization with {args.cluster_config}...")
            if not os.path.exists(args.cluster_config):
                raise FileNotFoundError(f"Cluster config not found: {args.cluster_config}")
                
            devices, links, rragc_config, layer_task, model_config = load_cluster_config(args.cluster_config)
            
            # Run Init Algorithm
            init_result = run_initialization(devices, links, rragc_config, layer_task, model_config)
            
            # Parse Result
            pipeline_order = init_result.rragc_result.pipeline_order
            vg_map = init_result.rragc_result.device_to_vg_map
            layer_alloc = init_result.intervg_layer_allocation
            
            # Construct stage_ranks
            stage_ranks = []
            layers_per_stage = []
            
            for i, vg_id in enumerate(pipeline_order):
                members = sorted([d_id for d_id, v in vg_map.items() if v == vg_id])
                if not members: continue
                stage_ranks.append(members)
                
                # Layer allocation
                if i < len(layer_alloc):
                    layers_per_stage.append(layer_alloc[i])
                else:
                    layers_per_stage.append(0)
            
            config_data = {
                "stage_ranks": stage_ranks,
                "layers_per_stage": layers_per_stage,
                "model_config": {
                    "n_layers": model_config.total_layers,
                    # Simplified mapping
                }
            }
            
            print(f"[Rank 0] Auto Configuration Result:")
            print(f"  Stage Ranks: {stage_ranks}")
            print(f"  Layers per Stage: {layers_per_stage}")
            
        else:
            # Manual Mode
            print(f"[Rank 0] Running Manual Configuration...")
            if not args.manual_config:
                raise ValueError("Manual mode requires --manual_config")
                
            stage_ranks = json.loads(args.manual_config)
            # Default even split
            n_stages = len(stage_ranks)
            total_layers = 24 # Default
            layers_per_stage = [total_layers // n_stages] * n_stages
            remaining = total_layers % n_stages
            for i in range(remaining):
                layers_per_stage[i] += 1
                
            config_data = {
                "stage_ranks": stage_ranks,
                "layers_per_stage": layers_per_stage,
                "model_config": {"n_layers": total_layers}
            }
            
        # Serialize
        serialized_config = pickle.dumps(config_data)
        config_size = torch.tensor([len(serialized_config)], dtype=torch.long)
    else:
        config_size = torch.tensor([0], dtype=torch.long)
        
    # Broadcast Config
    dist.broadcast(config_size, src=0)
    
    if rank != 0:
        serialized_config_tensor = torch.empty(config_size.item(), dtype=torch.uint8)
    else:
        serialized_config_tensor = torch.from_numpy(np.frombuffer(serialized_config, dtype=np.uint8)).clone() if 'np' in locals() else torch.ByteTensor(list(serialized_config))

    # Note: torch.ByteTensor(list(serialized_config)) is slow. 
    # Better to use ByteTensor of size, then copy.
    if rank == 0:
        # Create tensor from bytes
        # Using a byte tensor
        serialized_config_tensor = torch.tensor(list(serialized_config), dtype=torch.uint8)
        
    dist.broadcast(serialized_config_tensor, src=0)
    
    if rank != 0:
        config_data = pickle.loads(serialized_config_tensor.numpy().tobytes())
        
    # 3. Initialize Model
    dist_config = DistributedConfig(n_layers=config_data["model_config"]["n_layers"])
    stage_ranks = config_data["stage_ranks"]
    layers_per_stage = config_data["layers_per_stage"]
    
    # Check if I am in the ranks
    my_rank_in_config = False
    for ranks in stage_ranks:
        if rank in ranks:
            my_rank_in_config = True
            break
            
    if not my_rank_in_config:
        print(f"[Rank {rank}] Not assigned to any stage. Idle.")
        return

    model = DistributedQwen3Model(dist_config, stage_ranks, layers_per_stage, rank, world_size)
    
    # 4. Run Inference (Mock)
    if rank == stage_ranks[0][0]: # Input at first stage leader
        print(f"[Rank {rank}] Starting Inference...")
        dummy_input = torch.randint(0, dist_config.vocab_size, (1, 32)) # Batch 1, Seq 32
        output = model(dummy_input)
        # Note: output is None for stage 0 if it's not the last stage
    else:
        output = model(None) # Wait for input
        
    if rank == stage_ranks[-1][0]:
        print(f"[Rank {rank}] Final Output Shape: {output.shape}")

if __name__ == "__main__":
    run_distributed_inference()
