import os
import sys
import json
import argparse
import socket
import time
import pickle
import torch
import torch.nn as nn
try:
    import torch.distributed as dist
    HAS_DISTRIBUTED = hasattr(dist, 'init_process_group')
except ImportError:
    import torch.distributed as dist
    HAS_DISTRIBUTED = False
import torch.multiprocessing as mp
from safetensors.torch import load_file
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Any
import logging

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Profiler ---
class InferenceProfiler:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.records = {}  # layer_idx -> {'compute': 0.0, 'comm': 0.0, 'wait': 0.0}
        self.total_compute = 0.0
        self.total_comm = 0.0
        self.total_wait = 0.0
        
        # Temporary state for current layer
        self._layer_start_time = 0.0
        self._layer_comm_time = 0.0
        self._layer_wait_time = 0.0
        self._current_layer_idx = None

    def start_layer(self, layer_idx):
        self._current_layer_idx = layer_idx
        self._layer_comm_time = 0.0
        self._layer_wait_time = 0.0
        self._layer_start_time = time.perf_counter()
        if layer_idx not in self.records:
            self.records[layer_idx] = {'compute': 0.0, 'comm': 0.0, 'wait': 0.0}

    def end_layer(self):
        if self._current_layer_idx is None:
            return
            
        total_duration = time.perf_counter() - self._layer_start_time
        # Compute time = Total time - Communication time - Wait time
        compute_duration = max(0.0, total_duration - self._layer_comm_time - self._layer_wait_time)
        
        idx = self._current_layer_idx
        self.records[idx]['compute'] += compute_duration
        self.records[idx]['comm'] += self._layer_comm_time
        self.records[idx]['wait'] += self._layer_wait_time
        
        self.total_compute += compute_duration
        # total_comm and total_wait are updated via record methods
        
        self._current_layer_idx = None

    def record_comm(self, duration):
        self.total_comm += duration
        if self._current_layer_idx is not None:
            self._layer_comm_time += duration

    def record_compute(self, duration):
        self.total_compute += duration

    def record_wait(self, duration):
        self.total_wait += duration
        if self._current_layer_idx is not None:
            self._layer_wait_time += duration
            
    def print_stats(self, rank):
        print(f"\n[Rank {rank}] === Inference Performance Stats ===")
        print(f"{'Layer':<10} | {'Compute (s)':<15} | {'Comm (s)':<15} | {'Wait (s)':<15} | {'Total (s)':<15}")
        print("-" * 85)
        
        sorted_layers = sorted(self.records.keys())
        for idx in sorted_layers:
            comp = self.records[idx]['compute']
            comm = self.records[idx]['comm']
            wait = self.records[idx]['wait']
            total = comp + comm + wait
            print(f"{idx:<10} | {comp:<15.6f} | {comm:<15.6f} | {wait:<15.6f} | {total:<15.6f}")
            
        print("-" * 85)
        print(f"Total Compute: {self.total_compute:.6f} s")
        print(f"Total Comm:    {self.total_comm:.6f} s")
        print(f"Total Wait:    {self.total_wait:.6f} s")
        print(f"Total Time:    {self.total_compute + self.total_comm + self.total_wait:.6f} s")
        print("=" * 85 + "\n")

# --- Model Components ---

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
    def __init__(self, cfg, tp_rank, tp_world_size, tp_group=None, tp_ratios=None, profiler=None):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size
        self.tp_group = tp_group
        self.profiler = profiler
        self.hidden_dim = cfg.hidden_dim
        self.emb_dim = cfg.emb_dim
        
        # Non-uniform splitting logic
        if tp_ratios is None:
            tp_ratios = [1] * tp_world_size
        
        total_ratio = sum(tp_ratios)
        # Calculate start and end indices based on cumulative ratios to ensure full coverage
        # even if division is not exact (approximate handling for FFN)
        start_ratio = sum(tp_ratios[:tp_rank])
        end_ratio = sum(tp_ratios[:tp_rank + 1])
        
        start_idx = int(round(start_ratio * self.hidden_dim / total_ratio))
        end_idx = int(round(end_ratio * self.hidden_dim / total_ratio))
        
        self.local_hidden_dim = end_idx - start_idx
        
        self.fc1 = nn.Linear(cfg.emb_dim, self.local_hidden_dim, dtype=cfg.dtype, bias=False)
        self.fc2 = nn.Linear(cfg.emb_dim, self.local_hidden_dim, dtype=cfg.dtype, bias=False)
        self.fc3 = nn.Linear(self.local_hidden_dim, cfg.emb_dim, dtype=cfg.dtype, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x_intermediate = nn.functional.silu(x_fc1) * x_fc2
        x_out = self.fc3(x_intermediate)
        
        if self.tp_world_size > 1:
            t0 = time.perf_counter()
            # Async AllReduce
            handle = dist.all_reduce(x_out, op=dist.ReduceOp.SUM, group=self.tp_group, async_op=True)
            # Wait for completion
            handle.wait()
            if self.profiler:
                self.profiler.record_comm(time.perf_counter() - t0)
            
        return x_out

class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg, tp_rank, tp_world_size, tp_group=None, tp_ratios=None, profiler=None):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size
        self.tp_group = tp_group
        self.profiler = profiler
        self.num_heads = cfg.n_heads
        self.num_kv_groups = cfg.n_kv_groups
        self.head_dim = cfg.head_dim
        
        # Non-uniform splitting logic
        if tp_ratios is None:
            tp_ratios = [1] * tp_world_size
            
        total_ratio = sum(tp_ratios)
        start_ratio = sum(tp_ratios[:tp_rank])
        end_ratio = sum(tp_ratios[:tp_rank + 1])
        
        # Calculate local heads
        start_head_idx = int(round(start_ratio * self.num_heads / total_ratio))
        end_head_idx = int(round(end_ratio * self.num_heads / total_ratio))
        self.local_num_heads = end_head_idx - start_head_idx
        
        # Calculate needed KV heads logic
        # Each query head h maps to kv head h // group_size
        group_size = self.num_heads // self.num_kv_groups
        needed_kv_indices = set()
        self.q_head_to_kv_head = [] # Map local q index to local kv index
        
        # Find all unique needed KV indices
        for h in range(start_head_idx, end_head_idx):
            kv_idx = h // group_size
            needed_kv_indices.add(kv_idx)
            
        self.kv_head_indices = sorted(list(needed_kv_indices))
        self.local_kv_groups = len(self.kv_head_indices)
        
        # Create map from local Q index to local KV index
        global_kv_to_local = {k: i for i, k in enumerate(self.kv_head_indices)}
        for h in range(start_head_idx, end_head_idx):
            global_kv = h // group_size
            self.q_head_to_kv_head.append(global_kv_to_local[global_kv])
            
        self.q_head_to_kv_head = torch.tensor(self.q_head_to_kv_head, dtype=torch.long)
        
        self.d_out = self.local_num_heads * self.head_dim
        self.d_kv = self.local_kv_groups * self.head_dim
        
        self.W_query = nn.Linear(cfg.emb_dim, self.d_out, bias=True, dtype=cfg.dtype)
        self.W_key = nn.Linear(cfg.emb_dim, self.d_kv, bias=True, dtype=cfg.dtype)
        self.W_value = nn.Linear(cfg.emb_dim, self.d_kv, bias=True, dtype=cfg.dtype)
        self.out_proj = nn.Linear(self.d_out, cfg.emb_dim, bias=False, dtype=cfg.dtype)
        
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6) if cfg.qk_norm else None
        
        self.k_cache = {}
        self.v_cache = {}

    def forward(self, x, mask, cos, sin, start_pos=0, task_id=0):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        queries = queries.view(b, num_tokens, self.local_num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.local_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.local_kv_groups, self.head_dim).transpose(1, 2)
        
        if self.q_norm: queries = self.q_norm(queries)
        if self.k_norm: keys = self.k_norm(keys)
        
        # Apply RoPE to current keys and queries
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys = apply_rope(keys, cos, sin, offset=start_pos)
        
        # Update KV Cache
        if task_id not in self.k_cache:
            self.k_cache[task_id] = None
            self.v_cache[task_id] = None
            
        k_cache = self.k_cache[task_id]
        v_cache = self.v_cache[task_id]

        if start_pos == 0:
            # First step: initialize cache
            k_cache = keys
            v_cache = values
        else:
            # Subsequent steps: append to cache
            # Note: We need to handle the case where start_pos resets (new generation)
            # A simple way is to check if start_pos matches cache length
            if k_cache is not None and start_pos == k_cache.shape[2]:
                k_cache = torch.cat([k_cache, keys], dim=2)
                v_cache = torch.cat([v_cache, values], dim=2)
            else:
                # Reset or mismatch, re-initialize (should ideally not happen in correct flow)
                # But for safety, if start_pos < cache size, we truncate or reset.
                # Here we assume standard generation flow.
                if k_cache is not None and start_pos < k_cache.shape[2]:
                     k_cache = k_cache[:, :, :start_pos, :]
                     v_cache = v_cache[:, :, :start_pos, :]
                     k_cache = torch.cat([k_cache, keys], dim=2)
                     v_cache = torch.cat([v_cache, values], dim=2)
                else:
                    k_cache = keys
                    v_cache = values
        
        self.k_cache[task_id] = k_cache
        self.v_cache[task_id] = v_cache
        
        # Use cached keys/values for attention
        # keys/values: [b, n_kv_groups, seq_len, head_dim]
        keys = k_cache
        values = v_cache
        
        # Expand keys/values to match queries using the pre-calculated mapping
        # self.q_head_to_kv_head maps each local query head to the corresponding local KV head index
        if self.local_num_heads > 0:
            indices = self.q_head_to_kv_head.to(keys.device)
            keys = keys.index_select(1, indices)
            values = values.index_select(1, indices)
        
        attn_scores = queries @ keys.transpose(2, 3)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        out = self.out_proj(context)
        
        if self.tp_world_size > 1:
            t0 = time.perf_counter()
            # Async AllReduce
            handle = dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group, async_op=True)
            # Wait for completion
            handle.wait()
            if self.profiler:
                self.profiler.record_comm(time.perf_counter() - t0)
            
        return out

class TransformerBlock(nn.Module):
    def __init__(self, cfg, tp_rank, tp_world_size, tp_group=None, tp_ratios=None, profiler=None):
        super().__init__()
        self.att = GroupedQueryAttention(cfg, tp_rank, tp_world_size, tp_group, tp_ratios, profiler)
        self.ff = FeedForward(cfg, tp_rank, tp_world_size, tp_group, tp_ratios, profiler)
        self.norm1 = RMSNorm(cfg.emb_dim, eps=1e-6)
        self.norm2 = RMSNorm(cfg.emb_dim, eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0, task_id=0):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin, start_pos, task_id=task_id) + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x) + shortcut
        return x

class DistributedConfig:
    def __init__(self, vocab_size=151936, emb_dim=896, n_layers=24, n_heads=14, n_kv_groups=2, head_dim=64, hidden_dim=4864, rope_base=1000000, context_length=32768, dtype=torch.float32, qk_norm=False):
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

class CommManager:
    """Helper class for managing communication buffers and optimized operations"""
    def __init__(self, hidden_dim, dtype=torch.float32, device='cpu'):
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.device = device
        # Buffer for Decode Step (batch=1, seq=1): 
        # Structure: [batch(1), seq(1), hidden_dim, start_pos(1), task_id(1), data(1*1*hidden_dim)]
        # Total size: 5 + hidden_dim floats
        self.decode_buffer_size = 5 + hidden_dim
        # Map task_id -> buffer
        self.decode_buffers = {}
        # Map task_id -> signal buffer
        self.signal_buffers = {}
        # Map task_id -> active send request (list or single)
        self.send_reqs = {}
        
    def pack_decode(self, x: torch.Tensor, start_pos: int, task_id: int):
        """Pack (1, 1, hidden_dim) tensor and start_pos into decode_buffer for specific task"""
        # x shape should be (1, 1, hidden_dim)
        if x.shape[0] != 1 or x.shape[1] != 1:
             raise ValueError(f"CommManager.pack_decode only supports shape (1, 1, D), got {x.shape}")
        
        # Ensure previous send is complete before overwriting
        self.ensure_send_complete(task_id)

        # Ensure buffer exists
        if task_id not in self.decode_buffers:
            self.decode_buffers[task_id] = torch.zeros(self.decode_buffer_size, dtype=self.dtype, device=self.device)
            
        buffer = self.decode_buffers[task_id]
             
        # Use flat view for efficiency
        # Metadata: 0:batch, 1:seq, 2:dim, 3:start_pos, 4:task_id
        buffer[0] = 1.0
        buffer[1] = 1.0
        buffer[2] = float(self.hidden_dim)
        buffer[3] = float(start_pos)
        buffer[4] = float(task_id)
        
        # Data: 5:end
        buffer[5:].copy_(x.flatten())
        return buffer

    def get_signal_buffer(self, task_id):
        if task_id not in self.signal_buffers:
            self.signal_buffers[task_id] = torch.zeros(1, dtype=torch.float64, device=self.device)
        return self.signal_buffers[task_id]

    def register_send_req(self, task_id, req):
        """Register an active send request (or list of requests) for a task."""
        # Wait for previous if any (should have been called by ensure, but for safety)
        self.ensure_send_complete(task_id)
        self.send_reqs[task_id] = req

    def ensure_send_complete(self, task_id):
        """Ensure the last send for this task is complete."""
        if task_id in self.send_reqs and self.send_reqs[task_id] is not None:
            reqs = self.send_reqs[task_id]
            if isinstance(reqs, list):
                for r in reqs: r.wait()
            else:
                reqs.wait()
            self.send_reqs[task_id] = None

    def get_recv_buffer(self, task_id):
        """Get or create a receive buffer for specific task"""
        if task_id not in self.decode_buffers:
            self.decode_buffers[task_id] = torch.zeros(self.decode_buffer_size, dtype=self.dtype, device=self.device)
        return self.decode_buffers[task_id]

    def unpack_decode(self, task_id):
        """Unpack decode_buffer for specific task into x and start_pos"""
        if task_id not in self.decode_buffers:
             raise ValueError(f"No buffer for task {task_id}")
             
        buffer = self.decode_buffers[task_id]
        
        # Metadata
        b = int(buffer[0].item())
        s = int(buffer[1].item())
        d = int(buffer[2].item())
        start_pos = int(buffer[3].item())
        tid = int(buffer[4].item())
        
        if tid != task_id:
            # This should not happen if we used the correct buffer for recv
            pass
        
        # Data
        x = buffer[5:].view(b, s, d).clone()
        return x, start_pos, tid

class StaticDistributedQwen3Model(nn.Module):
    def __init__(self, config, my_config, tp_group=None):
        super().__init__()
        self.config = config
        self.tp_group = tp_group
        self.profiler = InferenceProfiler()
        self.my_rank = my_config['my_rank']
        self.world_size = my_config['world_size']
        
        # Initialize Communication Manager
        self.comm_manager = CommManager(config.emb_dim, dtype=config.dtype)
        self.my_stage_idx = my_config['my_stage_idx']
        self.stage_ranks = my_config['stage_ranks']
        self.layers_per_stage = my_config['layers_per_stage']
        
        self.tp_group_ranks = self.stage_ranks[self.my_stage_idx]
        self.tp_world_size = len(self.tp_group_ranks)
        self.tp_rank = self.tp_group_ranks.index(self.my_rank)
        
        self.tp_ratios = my_config.get('tp_ratios', [1] * self.tp_world_size)
        
        self.start_layer = sum(self.layers_per_stage[:self.my_stage_idx])
        self.num_layers = self.layers_per_stage[self.my_stage_idx]
        
        logger.info(f"[Rank {self.my_rank}] Stage {self.my_stage_idx}, TP {self.tp_rank}/{self.tp_world_size}, Layers {self.start_layer}-{self.start_layer+self.num_layers-1}, Ratios: {self.tp_ratios}")
        
        self.layers = nn.ModuleList([
            TransformerBlock(config, self.tp_rank, self.tp_world_size, self.tp_group, self.tp_ratios, self.profiler) 
            for _ in range(self.num_layers)
        ])
        
        cos, sin = compute_rope_params(config.head_dim, theta_base=config.rope_base, context_length=config.context_length, dtype=torch.float32)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
        if self.my_stage_idx == 0:
            self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim, dtype=config.dtype)
        
        if self.my_stage_idx == len(self.stage_ranks) - 1:
            self.final_norm = RMSNorm(config.emb_dim)
            self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False, dtype=config.dtype)

    def load_weights(self, model_path):
        logger.info(f"[Rank {self.my_rank}] Loading weights...")
        state_dict = load_file(model_path)
        
        def load_layer_weights(layer_idx, block):
            prefix = f"model.layers.{layer_idx}."
            
            def load_split(target, key, dim):
                w = state_dict[key]
                total_ratio = sum(self.tp_ratios)
                dim_size = w.shape[dim]
                size_per_ratio = dim_size / total_ratio
                
                # Calculate current rank split position
                start = int(round(sum(self.tp_ratios[:self.tp_rank]) * size_per_ratio))
                end = int(round(sum(self.tp_ratios[:self.tp_rank + 1]) * size_per_ratio))
                
                if dim == 0: target.data.copy_(w[start:end])
                else: target.data.copy_(w[:, start:end])

            def load_split_heads(target, key, dim, num_groups, head_dim, indices=None):
                w = state_dict[key]
                total_ratio = sum(self.tp_ratios)
                
                if indices is None:
                    # Calculate start/end based on GROUPS to match __init__ logic
                    start_ratio = sum(self.tp_ratios[:self.tp_rank])
                    end_ratio = sum(self.tp_ratios[:self.tp_rank + 1])
                    
                    start_group = int(round(start_ratio * num_groups / total_ratio))
                    end_group = int(round(end_ratio * num_groups / total_ratio))
                    
                    start_idx = start_group * head_dim
                    end_idx = end_group * head_dim
                    
                    # If target is empty (local_groups=0), skip copy
                    if start_idx == end_idx:
                        return

                    if dim == 0: target.data.copy_(w[start_idx:end_idx])
                    else: target.data.copy_(w[:, start_idx:end_idx])
                else:
                    # Load specific indices
                    if len(indices) == 0:
                        return
                        
                    slices = []
                    for idx in indices:
                        start = idx * head_dim
                        end = (idx + 1) * head_dim
                        if dim == 0:
                            slices.append(w[start:end])
                        else:
                            slices.append(w[:, start:end])
                    
                    if dim == 0:
                        target.data.copy_(torch.cat(slices, dim=0))
                    else:
                        target.data.copy_(torch.cat(slices, dim=1))

            load_split_heads(block.att.W_query.weight, f"{prefix}self_attn.q_proj.weight", 0, self.config.n_heads, self.config.head_dim)
            load_split_heads(block.att.W_query.bias, f"{prefix}self_attn.q_proj.bias", 0, self.config.n_heads, self.config.head_dim)
            load_split_heads(block.att.W_key.weight, f"{prefix}self_attn.k_proj.weight", 0, self.config.n_kv_groups, self.config.head_dim, indices=block.att.kv_head_indices)
            load_split_heads(block.att.W_key.bias, f"{prefix}self_attn.k_proj.bias", 0, self.config.n_kv_groups, self.config.head_dim, indices=block.att.kv_head_indices)
            load_split_heads(block.att.W_value.weight, f"{prefix}self_attn.v_proj.weight", 0, self.config.n_kv_groups, self.config.head_dim, indices=block.att.kv_head_indices)
            load_split_heads(block.att.W_value.bias, f"{prefix}self_attn.v_proj.bias", 0, self.config.n_kv_groups, self.config.head_dim, indices=block.att.kv_head_indices)
            load_split_heads(block.att.out_proj.weight, f"{prefix}self_attn.o_proj.weight", 1, self.config.n_heads, self.config.head_dim)
            
            load_split(block.ff.fc1.weight, f"{prefix}mlp.gate_proj.weight", 0)
            load_split(block.ff.fc2.weight, f"{prefix}mlp.up_proj.weight", 0)
            load_split(block.ff.fc3.weight, f"{prefix}mlp.down_proj.weight", 1)
            
            block.norm1.scale.data.copy_(state_dict[f"{prefix}input_layernorm.weight"])
            block.norm2.scale.data.copy_(state_dict[f"{prefix}post_attention_layernorm.weight"])

        if self.my_stage_idx == 0:
            self.tok_emb.weight.data.copy_(state_dict["model.embed_tokens.weight"])
            
        for i, layer in enumerate(self.layers):
            load_layer_weights(self.start_layer + i, layer)
            
        if self.my_stage_idx == len(self.stage_ranks) - 1:
            self.final_norm.scale.data.copy_(state_dict["model.norm.weight"])
            # Load full output head weights (no splitting)
            if "lm_head.weight" in state_dict:
                self.out_head.weight.data.copy_(state_dict["lm_head.weight"])
            else:
                self.out_head.weight.data.copy_(state_dict["model.embed_tokens.weight"])
        logger.info(f"[Rank {self.my_rank}] Weights loaded.")

    def forward(self, x, start_pos=0, task_id=0):
        # Stage > 0: Receive from Prev Stage
        if self.my_stage_idx > 0:
            t0 = time.perf_counter()
            prev_root = self.stage_ranks[self.my_stage_idx - 1][0]
            if self.tp_rank == 0:
                # Optimized for decode step (seq_len=1)
                # We assume decode if shape is (1, 1, hidden) but here we don't know shape yet.
                # To support optimization, we can first receive a small "flag" or assume decode.
                # However, for simplicity and robustness, we can try to receive the fixed-size decode buffer first?
                # No, standard send/recv must match size.
                # Since this is "Static" inference, we can assume:
                # If we are in "decode mode" (which caller knows?), we use optimized path.
                # But here `forward` doesn't know if it's decode or prefill easily without extra args.
                # Heuristic: We can check `x` shape if we are sender. But we are receiver here.
                # Let's rely on the fact that for LLM inference, prefill is rare (once) and decode is frequent.
                # We can add a `is_prefill` flag to forward? Or just check start_pos?
                # start_pos=0 usually implies prefill, but not always (could be single token input).
                
                # BETTER APPROACH:
                # The sender (Stage i-1) decides.
                # But receiver needs to know what to receive.
                # Let's add `use_optimized_comm` argument to forward, defaulting to True for single token?
                # Or just assume `x.shape[1] == 1` implies optimized path?
                # Receiver doesn't know x.shape[1].
                
                # Protocol Change:
                # Always receive a fixed header? No, latency.
                # Solution: The caller (inference loop) knows.
                # We can add `is_decode` arg to forward.
                pass

            # Since I cannot change the signature of forward() easily without updating all calls (which is fine),
            # I will use `x` (input) to determine mode? 
            # Wait, Stage > 0 `x` is None or dummy at start.
            
            # Let's assume:
            # If `start_pos > 0`, it is likely decode.
            # If `start_pos == 0`, it is prefill.
            # But what if we resume?
            # Let's just use the `x` argument passed to `forward`.
            # In `static_inference.py` main loop:
            # Stage 0 calls `model(input_ids, start_pos)`.
            # Other Stages call `model(None, start_pos=0)`?
            # Actually, `start_pos` is passed to all stages in the loop?
            # In the main loop:
            # `logits = model(input_ids, start_pos=start_pos)`
            # All ranks call this. `start_pos` is synced (incremented locally).
            # So all ranks know `start_pos`.
            # If `start_pos > 0`, we can assume decode (seq_len=1).
            # If `start_pos == 0`, it is prefill (seq_len >= 1).
            
            # Wait, `start_pos` starts at 0.
            # Prefill: `start_pos=0`, `seq_len` = input length.
            # Decode step 1: `start_pos=input_length`, `seq_len=1`.
            # So if `start_pos > 0`, it is definitely decode (seq_len=1) in standard autoregressive.
            # What if input_length=1? Then start_pos=0 and seq_len=1.
            # In this case, we use the "prefill" path (3-step) which is fine for the very first token.
            # Subsequent tokens will have start_pos > 0.
            
            # Logic:
            # if start_pos > 0: use optimized decode comm (1 call).
            # else: use robust comm (3 calls).
            
            if start_pos > 0:
                # Optimized Path (Decode)
                if self.tp_rank == 0:
                    t0 = time.perf_counter()
                    # 1. Wait for Signal (Wait Time)
                    signal = torch.zeros(1, dtype=torch.float64)
                    dist.recv(signal, src=prev_root)
                    t1 = time.perf_counter()
                    self.profiler.record_wait(t1 - t0)

                    # 2. Receive Data (Transfer Time)
                    # Use specific buffer for this task
                    recv_buf = self.comm_manager.get_recv_buffer(task_id)
                    dist.recv(recv_buf, src=prev_root)
                    
                    # Unpack
                    x_val, sp_val, tid_val = self.comm_manager.unpack_decode(task_id)
                    
                    if tid_val != task_id:
                        logger.warning(f"[Rank {self.my_rank}] Task ID mismatch in optimized recv: expected {task_id}, got {tid_val}")
                    
                    x = x_val
                    # start_pos is updated from message
                    start_pos = sp_val
                    # task_id = tid_val # Trust local task_id from loop
                    
                    t2 = time.perf_counter()
                    self.profiler.record_comm(t2 - t1)
                else:
                    x = None
            else:
                # Robust Path (Prefill / Init)
                if self.tp_rank == 0:
                    t0 = time.perf_counter()
                    # 1. Wait for Signal (Wait Time)
                    signal = torch.zeros(1, dtype=torch.float64)
                    dist.recv(signal, src=prev_root)
                    t1 = time.perf_counter()
                    self.profiler.record_wait(t1 - t0)

                    # 2. Receive Data (Transfer Time)
                    shape = torch.zeros(3, dtype=torch.long)
                    dist.recv(shape, src=prev_root)
                    b, s, d = shape.tolist()
                    x = torch.zeros((b, s, d), dtype=self.config.dtype)
                    dist.recv(x, src=prev_root)
                    sp = torch.zeros(1, dtype=torch.long)
                    dist.recv(sp, src=prev_root)
                    start_pos = sp.item()
                    
                    tid = torch.zeros(1, dtype=torch.long)
                    dist.recv(tid, src=prev_root)
                    task_id = tid.item()
                    
                    t2 = time.perf_counter()
                    self.profiler.record_comm(t2 - t1)
                else:
                    x = None
            
            # Broadcast to TP group
            if self.tp_world_size > 1:
                t0 = time.perf_counter()
                if self.tp_rank == 0:
                    # We can also optimize TP broadcast?
                    # If start_pos > 0, we know shape is (1,1,D).
                    # We can broadcast data directly if we assume shape.
                    # But broadcast is cheap?
                    # Let's keep TP robust for now or apply similar logic.
                    shape = torch.tensor(x.shape, dtype=torch.long)
                    sp = torch.tensor([start_pos], dtype=torch.long)
                    tid = torch.tensor([task_id], dtype=torch.long)
                else:
                    shape = torch.zeros(3, dtype=torch.long)
                    sp = torch.zeros(1, dtype=torch.long)
                    tid = torch.zeros(1, dtype=torch.long)
                
                dist.broadcast(shape, src=self.stage_ranks[self.my_stage_idx][0], group=self.tp_group)
                if self.tp_rank != 0:
                    b, s, d = shape.tolist()
                    x = torch.zeros((b, s, d), dtype=self.config.dtype)
                dist.broadcast(x, src=self.stage_ranks[self.my_stage_idx][0], group=self.tp_group)
                dist.broadcast(sp, src=self.stage_ranks[self.my_stage_idx][0], group=self.tp_group)
                dist.broadcast(tid, src=self.stage_ranks[self.my_stage_idx][0], group=self.tp_group)
                start_pos = sp.item()
                task_id = tid.item()
                self.profiler.record_comm(time.perf_counter() - t0)

        # Stage 0: Embeddings
        if self.my_stage_idx == 0:
            # ... (Existing TP broadcast logic for Stage 0) ...
            if self.tp_world_size > 1:
                t0 = time.perf_counter()
                src_rank = self.stage_ranks[0][0]
                if self.tp_rank == 0:
                    shape = torch.tensor(x.shape, dtype=torch.long)
                    dist.broadcast(shape, src=src_rank, group=self.tp_group)
                    dist.broadcast(x, src=src_rank, group=self.tp_group)
                else:
                    shape = torch.zeros(2, dtype=torch.long)
                    dist.broadcast(shape, src=src_rank, group=self.tp_group)
                    x = torch.zeros(tuple(shape.tolist()), dtype=torch.long)
                    dist.broadcast(x, src=src_rank, group=self.tp_group)
                self.profiler.record_comm(time.perf_counter() - t0)
            
            t0 = time.perf_counter()
            x = self.tok_emb(x)
            self.profiler.record_compute(time.perf_counter() - t0)

        # Layers
        b, seq, _ = x.shape
        mask = None
        if seq > 1:
            mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1).to(x.device)
            
        for i, layer in enumerate(self.layers):
            layer_idx = self.start_layer + i
            self.profiler.start_layer(layer_idx)
            x = layer(x, mask, self.cos, self.sin, start_pos=start_pos, task_id=task_id)
            self.profiler.end_layer()
            
        # Send to Next Stage or Return Logits
        if self.my_stage_idx < len(self.stage_ranks) - 1:
            if self.tp_rank == 0:
                t0 = time.perf_counter()
                next_root = self.stage_ranks[self.my_stage_idx + 1][0]
                
                # Check condition for optimized send
                # Must match receiver's logic: start_pos > 0
                if start_pos > 0 and x.shape[0] == 1 and x.shape[1] == 1:
                    # Optimized Send (Async to enable pipeline overlap)
                    # 1. Send Signal
                    sig_buf = self.comm_manager.get_signal_buffer(task_id)
                    sig_buf[0] = time.perf_counter()
                    req_sig = dist.isend(sig_buf, dst=next_root)
                    
                    # 2. Send Data
                    packed_buf = self.comm_manager.pack_decode(x, start_pos, task_id)
                    req_data = dist.isend(packed_buf, dst=next_root)
                    
                    # Register requests to ensure completion before next reuse
                    self.comm_manager.register_send_req(task_id, [req_sig, req_data])
                else:
                    # Robust Send (Blocking)
                    # 1. Send Signal
                    signal = torch.tensor([time.perf_counter()], dtype=torch.float64)
                    dist.send(signal, dst=next_root)
                    
                    # 2. Send Data
                    dist.send(torch.tensor(x.shape, dtype=torch.long), dst=next_root)
                    dist.send(x, dst=next_root)
                    dist.send(torch.tensor([start_pos], dtype=torch.long), dst=next_root)
                    dist.send(torch.tensor([task_id], dtype=torch.long), dst=next_root)
                
                self.profiler.record_comm(time.perf_counter() - t0)
            
            return None
        else:
            t0 = time.perf_counter()
            x = self.final_norm(x)
            self.profiler.record_compute(time.perf_counter() - t0)
            
            if self.tp_world_size > 1:
                # If TP > 1, hidden states might be split or need aggregation if coming from a split layer
                # But here, TransformerBlock output is already aggregated if it was split (e.g. FFN output AllReduce)
                # However, if we split the Norm or something else, we might need synchronization.
                # In this implementation, FFN and Attn outputs are AllReduced within the block, 
                # so x entering final_norm is fully aggregated and identical across TP ranks.
                pass
            
            # Only Rank 0 of the last stage computes logits to save computation
            if self.tp_rank == 0:
                t0 = time.perf_counter()
                logits = self.out_head(x)
                self.profiler.record_compute(time.perf_counter() - t0)
                return logits
            return None

def parse_allocation_config(config_str: str, ip_list: List[str]) -> dict:
    stages_config = config_str.split('*')
    parsed_stages = []
    current_ip_idx = 0
    
    for s_conf in stages_config:
        ratios_part, layers_part = s_conf.split('@')
        num_layers = int(layers_part)
        ratios = [int(r) for r in ratios_part.split(':')]
        
        stage_devices = []
        for _ in ratios:
            if current_ip_idx >= len(ip_list): raise ValueError("Not enough IPs")
            stage_devices.append(ip_list[current_ip_idx])
            current_ip_idx += 1
            
        parsed_stages.append({
            "devices": stage_devices,
            "tp_ratios": ratios,
            "num_layers": num_layers,
            "root_ip": stage_devices[0]
        })
        
    return {
        "stages": parsed_stages,
        "global_root_ip": parsed_stages[0]["devices"][0],
        "total_layers": sum(s["num_layers"] for s in parsed_stages)
    }

def send_config_to_node(ip, port, config):
    try:
        if ':' in ip: ip, port = ip.split(':')[0], int(ip.split(':')[1])
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(30)
            logger.info(f"Sending config to {ip}:{port}")
            s.connect((ip, port))
            s.sendall(json.dumps(config).encode('utf-8'))
    except Exception as e:
        logger.error(f"Failed to send to {ip}: {e}")
        raise

def start_listener(ip, port):
    if ':' in ip: ip, port = ip.split(':')[0], int(ip.split(':')[1])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((ip, port))
        s.listen(1)
        logger.info(f"Listening on {ip}:{port}...")
        conn, addr = s.accept()
        with conn:
            logger.info(f"Connected by {addr}")
            data = conn.recv(1024 * 1024)
            if not data: return None
            return json.loads(data.decode('utf-8'))

class TaskState:
    def __init__(self, task_id, prompt, tokenizer=None, input_ids=None, device='cpu'):
        self.task_id = task_id
        self.prompt = prompt
        if input_ids is not None:
            self.input_ids = input_ids.to(device)
        elif tokenizer is not None:
             self.input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device).to(torch.long)
        else:
             self.input_ids = torch.empty(1, 1, dtype=torch.long, device=device) # Dummy for non-rank-0
             
        self.curr_input_ids = self.input_ids
        self.generated_ids = []
        if self.input_ids.numel() > 0:
            self.generated_ids = self.input_ids[0].tolist()
            
        self.start_pos = 0
        self.is_complete = False
        self.start_time = 0.0
        self.end_time = 0.0
        self.token_count = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["listen", "assign"])
    parser.add_argument("--my_ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument("--config", type=str, help="Allocation config string")
    parser.add_argument("--ips", type=str, help="Comma separated IPs")
    parser.add_argument("--model", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--prompts", type=str, help="Pipe-separated prompts")
    parser.add_argument("--prompts_file", type=str, help="File with prompts")
    parser.add_argument("--num_tasks", type=int, default=1)
    parser.add_argument("--test_2", action="store_true")
    parser.add_argument("--test_3", action="store_true")
    parser.add_argument("--test_4", action="store_true")
    parser.add_argument("--test_multitask", action="store_true")
    args = parser.parse_args()

    if args.test_multitask:
        test_multitask_2_devices()
        return

    if args.test_2:
        test_local_2_devices()
        return
    if args.test_4:
        test_local_4_devices()
        return
    if args.test_3:
        test_local_3_devices()
        return

    if not args.mode:
        parser.error("the following arguments are required: --mode")

    my_config = None
    if args.mode == "listen":
        my_config = start_listener(args.my_ip, args.port)
    elif args.mode == "assign":
        ip_list = args.ips.split(',')
        alloc = parse_allocation_config(args.config, ip_list)
        all_devices = []
        for stage in alloc["stages"]: all_devices.extend(stage["devices"])
        
        stage_ranks, current_rank = [], 0
        for stage in alloc["stages"]:
            ranks = list(range(current_rank, current_rank + len(stage["devices"])))
            stage_ranks.append(ranks)
            current_rank += len(stage["devices"])
            
        layers_per_stage = [s["num_layers"] for s in alloc["stages"]]
        tp_ratios_list = [s["tp_ratios"] for s in alloc["stages"]]
        
        global_rank = 0
        for stage_idx, stage in enumerate(alloc["stages"]):
            for dev_idx, device_ip in enumerate(stage["devices"]):
                node_config = {
                    "my_rank": global_rank,
                    "my_stage_idx": stage_idx,
                    "is_stage_root": (dev_idx == 0),
                    "stage_ranks": stage_ranks,
                    "layers_per_stage": layers_per_stage,
                    "tp_ratios": tp_ratios_list[stage_idx],
                    "model_path": args.model,
                    "master_addr": alloc["global_root_ip"].split(':')[0],
                    "master_port": 29500 if ':' not in alloc["global_root_ip"] else int(alloc["global_root_ip"].split(':')[1]),
                    "world_size": len(all_devices)
                }
                if global_rank == 0: my_config = node_config
                else: send_config_to_node(device_ip, args.port, node_config)
                global_rank += 1

    if not my_config:
        logger.error("Failed to get config")
        return

    os.environ['MASTER_ADDR'] = my_config['master_addr']
    os.environ['MASTER_PORT'] = str(my_config['master_port'])
    
    if HAS_DISTRIBUTED:
        dist.init_process_group(backend="gloo", rank=my_config['my_rank'], world_size=my_config['world_size'])
        
        # Create TP groups
        tp_group = None
        for ranks in my_config['stage_ranks']:
            ranks = sorted(ranks)
            if len(ranks) > 1:
                g = dist.new_group(ranks)
                if my_config['my_rank'] in ranks:
                    tp_group = g
    else:
        if my_config['world_size'] > 1:
            logger.error("Error: This PyTorch installation does not support distributed execution, but world_size > 1.")
            logger.error("Please install a PyTorch version with distributed support or run on a single device.")
            sys.exit(1)
        logger.warning("Warning: PyTorch distributed support missing. Running in single-device mode.")
        tp_group = None
    
    dist_config = DistributedConfig() # Use defaults or pass params
    model = StaticDistributedQwen3Model(dist_config, my_config, tp_group=tp_group)
    model.load_weights(my_config['model_path'])
    
    # --- Task Initialization ---
    tasks = []
    tokenizer = None
    
    if my_config['my_rank'] == 0:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        
        prompt_list = []
        if args.prompts:
            prompt_list = args.prompts.split('|')
        elif args.prompts_file:
            try:
                with open(args.prompts_file, 'r') as f:
                    prompt_list = [line.strip() for line in f if line.strip()]
            except Exception as e:
                logger.error(f"Failed to read prompts file: {e}")
                prompt_list = ["Hello"]
        elif args.prompt:
            prompt_list = [args.prompt]
        else:
            prompt_list = ["Hello"]
            
        # Ensure we have enough prompts for num_tasks by cycling
        if len(prompt_list) < args.num_tasks:
            import itertools
            prompt_list = list(itertools.islice(itertools.cycle(prompt_list), args.num_tasks))
        else:
            prompt_list = prompt_list[:args.num_tasks]
            
        print(f"\n[Rank 0] Initializing {args.num_tasks} tasks with prompts: {prompt_list}")
        
        for i, p in enumerate(prompt_list):
            tasks.append(TaskState(i, p, tokenizer=tokenizer))
    else:
        # Other ranks initialize dummy tasks
        for i in range(args.num_tasks):
            tasks.append(TaskState(i, "", input_ids=None))

    # Identify the rank that will broadcast the sampled token (Root of the last stage)
    last_stage_idx = len(my_config['stage_ranks']) - 1
    token_src_rank = my_config['stage_ranks'][last_stage_idx][0]

    model.profiler.reset()
    
    # --- Performance Metrics Initialization ---
    inference_start_time = time.perf_counter()
    
    # We generate 20 new tokens per task
    MAX_NEW_TOKENS = 20
    
    # Loop until all tasks are complete
    step_count = 0
    while True:
        # Check completion
        active_tasks = [t for t in tasks if not t.is_complete]
        if not active_tasks:
            break
            
        step_count += 1
        
        # 1. Forward Pass (Pipeline Filling)
        # Iterate over all active tasks to keep pipeline full
        step_logits = {} # task_id -> logits
        
        for task in tasks:
            if task.is_complete: continue
            
            # Start timing for task on first step
            if my_config['my_rank'] == 0 and task.token_count == 0 and task.start_time == 0.0:
                task.start_time = time.perf_counter()

            if my_config['my_rank'] == 0:
                # Stage 0: Feed input
                logits = model(task.curr_input_ids, start_pos=task.start_pos, task_id=task.task_id)
            else:
                # Other Stages: Receive
                logits = model(None, start_pos=task.start_pos, task_id=task.task_id)

            if my_config['my_rank'] == token_src_rank:
                step_logits[task.task_id] = logits

        # 2. Sample and Broadcast (Sync Point)
        # We must broadcast for ALL active tasks to keep ranks in sync
        for task in tasks:
            if task.is_complete: continue
            
            # Payload: [token, step_len, task_id, is_complete_flag]
            # Use fixed size tensor for broadcast
            payload = torch.zeros(4, dtype=torch.long)
            
            if my_config['my_rank'] == token_src_rank:
                if task.task_id not in step_logits:
                    logger.error(f"Missing logits for task {task.task_id}")
                    # Should crash or handle
                
                logits = step_logits[task.task_id]
                # logits: [batch, seq, vocab]
                next_token_val = torch.argmax(logits[:, -1, :], dim=-1).item()
                step_len = logits.shape[1]
                
                # Determine completion status BEFORE broadcast
                # Note: We haven't incremented token_count yet, so check +1
                is_task_complete = 0
                if task.token_count + 1 >= MAX_NEW_TOKENS:
                    is_task_complete = 1
                
                payload[0] = next_token_val
                payload[1] = step_len
                payload[2] = task.task_id
                payload[3] = is_task_complete
            
            if HAS_DISTRIBUTED:
                dist.broadcast(payload, src=token_src_rank)
            
            token_val = payload[0].item()
            step_len_val = payload[1].item()
            tid_val = payload[2].item()
            is_complete_val = payload[3].item()
            
            # Verify task_id
            if tid_val != task.task_id:
                # This could happen if ranks get out of sync on active tasks list
                # But since logic is deterministic, it should be fine.
                pass
                
            # Update Task State
            task.start_pos += step_len_val
            task.token_count += 1
            
            if is_complete_val == 1:
                task.is_complete = True
            
            if my_config['my_rank'] == 0:
                task.generated_ids.append(token_val)
                task.curr_input_ids = torch.tensor([[token_val]], dtype=torch.long)
                
                # Check completion
                if task.is_complete:
                    task.end_time = time.perf_counter()
                    
                    # Print Task Output immediately
                    print(f"\n[Rank 0] Task {task.task_id} Completed!")
                    print(f"Prompt: {task.prompt}")
                    output_text = tokenizer.decode(task.generated_ids)
                    print(f"Output: {output_text}")
                    duration = task.end_time - task.start_time
                    tps = task.token_count / duration if duration > 0 else 0
                    print(f"Duration: {duration:.2f}s, TPS: {tps:.2f}")
                    print("-" * 40)

    total_inference_time = time.perf_counter() - inference_start_time

    if my_config['my_rank'] == 0:
        print("\nAll tasks completed.")
            
    # model.profiler.print_stats(my_config['my_rank']) # Optional: Keep local stats or remove

    # --- Global Statistics Collection ---
    if HAS_DISTRIBUTED:
        # 1. Prepare Local Summary
        my_summary = {
            'rank': my_config['my_rank'],
            'stage_idx': my_config['my_stage_idx'],
            'layers_per_stage': my_config['layers_per_stage'], # List of layer counts per stage
            'records': model.profiler.records,
            'total_compute': model.profiler.total_compute,
            'total_comm': model.profiler.total_comm,
            'total_wait': model.profiler.total_wait,
        }
        
        # 2. Gather Summaries at Rank 0
        if my_config['my_rank'] == 0:
            all_summaries = [my_summary]
            # Receive from all other ranks
            for src in range(1, my_config['world_size']):
                # Receive size first
                size_tensor = torch.zeros(1, dtype=torch.long)
                dist.recv(size_tensor, src=src)
                data_size = size_tensor.item()
                
                # Receive data
                data_tensor = torch.zeros(data_size, dtype=torch.uint8)
                dist.recv(data_tensor, src=src)
                
                # Deserialize
                other_summary = pickle.loads(bytes(data_tensor.tolist()))
                all_summaries.append(other_summary)
            
            # 3. Process and Print Global Statistics
            print("="*60)
            print("Global Inference Statistics (Excludes Model Loading)")
            print("="*60)
            
            # Aggregate Throughput
            total_tokens = sum(t.token_count for t in tasks)
            throughput = total_tokens / total_inference_time
            print(f"Total Tokens Generated: {total_tokens}")
            print(f"Total Time: {total_inference_time:.4f}s")
            print(f"System Throughput: {throughput:.2f} tokens/s")
            print("-" * 60)
            
            # Global Layer Statistics
            # Combine records from all ranks
            global_layer_records = {}
            for summary in all_summaries:
                for layer_idx, record in summary['records'].items():
                    # If TP is used, multiple ranks might have the same layer.
                    # We should take the MAX compute and MAX comm? Or Average?
                    # For TP, compute is parallel, so max is the bottleneck.
                    # Comm is also parallel.
                    # However, simply taking the record from the first rank that has it is usually enough for homogeneous TP.
                    # Or better: average them if multiple ranks have the same layer.
                    if layer_idx not in global_layer_records:
                        global_layer_records[layer_idx] = {'compute': [], 'comm': [], 'wait': []}
                    global_layer_records[layer_idx]['compute'].append(record['compute'])
                    global_layer_records[layer_idx]['comm'].append(record['comm'])
                    global_layer_records[layer_idx]['wait'].append(record['wait'])
            
            print("Global Layer Statistics (Avg across TP ranks if applicable):")
            print(f"{'Layer':<10} | {'Compute (s)':<15} | {'Comm (s)':<15} | {'Wait (s)':<15} | {'Total (s)':<15}")
            print("-" * 85)
            
            sorted_layers = sorted(global_layer_records.keys())
            for idx in sorted_layers:
                comps = global_layer_records[idx]['compute']
                comms = global_layer_records[idx]['comm']
                waits = global_layer_records[idx]['wait']
                avg_comp = sum(comps) / len(comps)
                avg_comm = sum(comms) / len(comms)
                avg_wait = sum(waits) / len(waits)
                total = avg_comp + avg_comm + avg_wait
                print(f"{idx:<10} | {avg_comp:<15.6f} | {avg_comm:<15.6f} | {avg_wait:<15.6f} | {total:<15.6f}")
            print("-" * 85)

            # Stage Statistics
            print("[Per-Stage Summary]")
            print(f"{'Stage':<6} | {'Layers':<8} | {'Compute (s)':<12} | {'Comm (s)':<12} | {'Wait (s)':<12} | {'Total (s)':<12}")
            print("-" * 85)
            
            # Group summaries by stage
            stage_stats = {}
            for summary in all_summaries:
                s_idx = summary['stage_idx']
                if s_idx not in stage_stats:
                    stage_stats[s_idx] = {'compute': [], 'comm': [], 'wait': [], 'layers': 0}
                
                stage_stats[s_idx]['compute'].append(summary['total_compute'])
                stage_stats[s_idx]['comm'].append(summary['total_comm'])
                stage_stats[s_idx]['wait'].append(summary['total_wait'])
                stage_stats[s_idx]['layers'] = summary['layers_per_stage'][s_idx]

            sorted_stages = sorted(stage_stats.keys())
            
            # Calculate totals for bottleneck identification
            stage_totals = []
            for s_idx in sorted_stages:
                comps = stage_stats[s_idx]['compute']
                comms = stage_stats[s_idx]['comm']
                waits = stage_stats[s_idx]['wait']
                avg_comp = sum(comps) / len(comps)
                avg_comm = sum(comms) / len(comms)
                avg_wait = sum(waits) / len(waits)
                stage_totals.append((s_idx, avg_comp + avg_comm + avg_wait))
            
            # Find max total time
            max_total = max(t[1] for t in stage_totals) if stage_totals else 0
            
            for s_idx in sorted_stages:
                # Average across TP ranks in the same stage
                comps = stage_stats[s_idx]['compute']
                comms = stage_stats[s_idx]['comm']
                waits = stage_stats[s_idx]['wait']
                avg_comp = sum(comps) / len(comps)
                avg_comm = sum(comms) / len(comms)
                avg_wait = sum(waits) / len(waits)
                total = avg_comp + avg_comm + avg_wait
                layers_count = stage_stats[s_idx]['layers']
                
                bottleneck_mark = "(*)" if abs(total - max_total) < 1e-6 else ""
                
                print(f"{s_idx:<6} | {layers_count:<8} | {avg_comp:<12.4f} | {avg_comm:<12.4f} | {avg_wait:<12.4f} | {total:<12.4f} {bottleneck_mark}")
            print("=" * 85)
            print("(*) Indicates Bottleneck Stage")

        else:
            # Send Summary to Rank 0
            data_bytes = pickle.dumps(my_summary)
            data_tensor = torch.tensor(list(data_bytes), dtype=torch.uint8)
            size_tensor = torch.tensor([len(data_bytes)], dtype=torch.long)
            
            dist.send(size_tensor, dst=0)
            dist.send(data_tensor, dst=0)

    if HAS_DISTRIBUTED:
        dist.destroy_process_group()

def _run_test_process(mode, ip, port, config, ips, model_path, tok_path, num_tasks=1, prompts=None):
    sys.argv = ["static_inference.py", "--mode", mode, "--my_ip", ip, "--port", str(port), "--num_tasks", str(num_tasks)]
    if mode == "assign":
        sys.argv.extend(["--config", config, "--ips", ips, "--model", model_path, "--tokenizer", tok_path])
        if prompts:
            sys.argv.extend(["--prompts", prompts])
    main()

def test_multitask_2_devices():
    # 2 Devices, 3 Tasks
    model_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B/model.safetensors"
    tok_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B"
    # Use different ports (30600, 30601)
    ips = "127.0.0.1:30600,127.0.0.1:30601"
    config = "1@12*1@12"
    num_tasks = 3
    prompts = "Hello|How are you|What is AI"
    
    # Listener: num_tasks=3
    p1 = mp.Process(target=_run_test_process, args=("listen", "127.0.0.1", 30601, None, None, None, None, num_tasks))
    # Assigner: num_tasks=3, prompts provided
    p0 = mp.Process(target=_run_test_process, args=("assign", "127.0.0.1", 30600, config, ips, model_path, tok_path, num_tasks, prompts))
    
    p1.start()
    time.sleep(2)
    p0.start()
    p0.join()
    p1.join()

def test_local_2_devices():
    # 2 Devices: 1@12 * 1@12 (2 stages, no TP)
    model_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B/model.safetensors"
    tok_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B"
    # Use different ports (30500, 30501)
    ips = "127.0.0.1:30500,127.0.0.1:30501"
    config = "1@12*1@12"
    
    p1 = mp.Process(target=_run_test_process, args=("listen", "127.0.0.1", 30501, None, None, None, None))
    p0 = mp.Process(target=_run_test_process, args=("assign", "127.0.0.1", 30500, config, ips, model_path, tok_path))
    
    p1.start()
    time.sleep(2)
    p0.start()
    p0.join()
    p1.join()

def test_local_4_devices():
    # 4 Devices: 1:1@12 * 1:1@12 (2 stages, TP=2 each)
    model_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B/model.safetensors"
    tok_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B"
    # Use different ports to avoid conflict with test_2 (or previous runs)
    ips = "127.0.0.1:29600,127.0.0.1:29601,127.0.0.1:29602,127.0.0.1:29603"
    config = "1:1@12*1:1@12"
    
    procs = []
    # Listeners (Ports 29601, 29602, 29603)
    for port in [29601, 29602, 29603]:
        p = mp.Process(target=_run_test_process, args=("listen", "127.0.0.1", port, None, None, None, None))
        p.start()
        procs.append(p)
        
    time.sleep(2)
    # Assign (Port 29600)
    p0 = mp.Process(target=_run_test_process, args=("assign", "127.0.0.1", 29600, config, ips, model_path, tok_path))
    p0.start()
    procs.append(p0)
    
    for p in procs:
        p.join()

def test_local_3_devices():
    # 3 Devices: 5:1:1@24 (1 stage, non-uniform TP)
    model_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B/model.safetensors"
    tok_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B"
    ips = "127.0.0.1:29800,127.0.0.1:29801,127.0.0.1:29802"
    config = "5:1:1@24"
    
    procs = []
    # Listeners (Ports 29801, 29802)
    for port in [29801, 29802]:
        p = mp.Process(target=_run_test_process, args=("listen", "127.0.0.1", port, None, None, None, None))
        p.start()
        procs.append(p)
        
    time.sleep(2)
    # Assign (Port 29800)
    p0 = mp.Process(target=_run_test_process, args=("assign", "127.0.0.1", 29800, config, ips, model_path, tok_path))
    p0.start()
    procs.append(p0)
    
    for p in procs:
        p.join()

if __name__ == "__main__":
    main()
