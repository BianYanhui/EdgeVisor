
import os
import sys

# --- Fix for Jetson .local path ---
# Ensure local site-packages are included, especially for tiktoken/transformers
import site
user_site = site.getusersitepackages()
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

# Fallback for specific Python versions if site.getusersitepackages() is weird
local_site_packages = os.path.expanduser(f"~/.local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
if os.path.exists(local_site_packages) and local_site_packages not in sys.path:
    sys.path.insert(0, local_site_packages)

# Debug: Check imports
try:
    import tiktoken
    import sentencepiece
    # logger is not defined yet, use print
    # print(f"DEBUG: tiktoken found at {tiktoken.__file__}")
except ImportError as e:
    print(f"DEBUG: Import failed even after path fix: {e}")
    print(f"DEBUG: sys.path: {sys.path}")
# ----------------------------------

import json
import argparse
import socket
import time
import pickle
import struct
import mmap
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch.distributed as dist
    HAS_DISTRIBUTED = hasattr(dist, 'init_process_group')
except ImportError:
    import torch.distributed as dist
    HAS_DISTRIBUTED = False
import torch.multiprocessing as mp
from safetensors.torch import load_file
from safetensors import safe_open
import gc
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Any, Optional
import logging
from dataclasses import dataclass

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
@dataclass
class ModelConfig:
    dim: int = 1024
    hidden_dim: int = 3072
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 8
    vocab_size: int = 151936
    head_dim: int = 64
    rope_theta: float = 1000000.0
    norm_eps: float = 1e-6
    max_seq_len: int = 2048
    dtype: torch.dtype = torch.float32
    disable_tp: bool = False

# --- Profiler ---
class InferenceProfiler:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.records = {}
        self.total_compute = 0.0
        self.total_comm = 0.0
        self.total_wait = 0.0
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
        if self._current_layer_idx is None: return
        total_duration = time.perf_counter() - self._layer_start_time
        compute_duration = max(0.0, total_duration - self._layer_comm_time - self._layer_wait_time)
        idx = self._current_layer_idx
        self.records[idx]['compute'] += compute_duration
        self.records[idx]['comm'] += self._layer_comm_time
        self.records[idx]['wait'] += self._layer_wait_time
        self.total_compute += compute_duration
        self._current_layer_idx = None

    def record_comm(self, duration):
        self.total_comm += duration
        if self._current_layer_idx is not None: self._layer_comm_time += duration

    def record_compute(self, duration):
        self.total_compute += duration

    def record_wait(self, duration):
        self.total_wait += duration
        if self._current_layer_idx is not None: self._layer_wait_time += duration

    def print_stats(self):
        total_time = self.total_compute + self.total_comm + self.total_wait
        if total_time == 0: return
        
        print("-" * 60)
        print(f"Performance Report:")
        print(f"Total Time:     {total_time:.4f} s")
        print(f"Compute Time:   {self.total_compute:.4f} s ({self.total_compute/total_time*100:.1f}%)")
        print(f"Comm Time:      {self.total_comm:.4f} s ({self.total_comm/total_time*100:.1f}%)")
        print(f"Wait/Idle Time: {self.total_wait:.4f} s ({self.total_wait/total_time*100:.1f}%)")
        
        # Breakdown by layer type (first layer only as sample)
        first_layer = min(self.records.keys()) if self.records else -1
        if first_layer != -1:
            rec = self.records[first_layer]
            print(f"Sample Layer {first_layer}: Compute={rec['compute']*1000:.2f}ms, Comm={rec['comm']*1000:.2f}ms")
        print("-" * 60)

# --- Optimization Components ---

class QuantizedLinear(nn.Module):
    """
    Custom Linear layer that holds quantized weights (Int8).
    Performs quantized matrix multiplication using int8 inputs and weights.
    Supports 'int8' activation quantization on the fly.
    """
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        
        # Quantized storage
        self.register_buffer('weight_data', torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer('scale', torch.ones((out_features, 1), dtype=torch.float32))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: [batch, seq, in_features] (float32)
        
        # Fast path for CPU inference without int8 acceleration
        # If we don't have _int_mm, skipping activation quantization is FASTER
        # and avoids the overhead of per-token scaling.
        # We just dequantize weights block-wise (or row-wise) and use float matmul.
        
        if hasattr(torch, '_int_mm'):
             # ... (Keep existing _int_mm logic)
            x_shape = x.shape
            if len(x_shape) == 3:
                x_flat = x.view(-1, self.in_features)
            else:
                x_flat = x
                
            # Calculate scale per token: [B*S, 1]
            abs_x = x_flat.abs()
            max_val = abs_x.max(dim=1, keepdim=True).values
            scale_x = max_val / 127.0
            scale_x = torch.where(scale_x == 0, torch.tensor(1.0, device=x.device, dtype=x.dtype), scale_x)
            
            # Quantize
            x_int8 = (x_flat / scale_x).round().clamp(-127, 127).to(torch.int8)
            y_int32 = torch._int_mm(x_int8, self.weight_data.t())
            
            y = y_int32.to(self.dtype) * scale_x * self.scale.view(1, -1)
            
            if len(x_shape) == 3:
                y = y.view(x_shape[0], x_shape[1], self.out_features)
                
            if self.bias is not None:
                y = y + self.bias
            return y

        else:
            # Fallback Optimization: Direct Float Matmul
            # Skip x quantization entirely.
            # Just dequantize weights on the fly and matmul.
            # This saves: abs, max, div, round, clamp, to(int8), and later dequant logic.
            
            # x: [batch, seq, in]
            # w_int8: [out, in]
            # scale: [out, 1]
            
            # We need w_float = w_int8 * scale
            # But w_float is too big to materialize fully (OOM).
            # So we chunk along OUT dimension.
            
            M = x.numel() // self.in_features
            K = self.in_features
            N = self.out_features
            
            x_flat = x.view(M, K)
            
            # Output buffer
            y = torch.empty((M, N), dtype=self.dtype, device=x.device)
            
            chunk_size = 512 # Tunable
            
            # Pre-transpose x for better memory access if M is large? 
            # Usually M=1 (decoding), so x is [1, K].
            
            for i in range(0, N, chunk_size):
                end_i = min(i + chunk_size, N)
                
                # Dequantize weight chunk
                # w_chunk_int8: [chunk, K]
                w_chunk_int8 = self.weight_data[i:end_i, :]
                scale_chunk = self.scale[i:end_i, :] # [chunk, 1]
                
                # w_chunk_float = w_chunk_int8 * scale_chunk
                w_chunk_float = w_chunk_int8.to(self.dtype) * scale_chunk
                
                # Matmul: [M, K] @ [chunk, K].T -> [M, chunk]
                # Using linear is slightly cleaner: y = x @ w.T
                # F.linear(input, weight, bias=None)
                # weight shape for F.linear is (out_features, in_features)
                
                # Check shapes:
                # x_flat: [M, K]
                # w_chunk_float: [chunk, K]
                # F.linear(x_flat, w_chunk_float) -> [M, chunk]
                
                y[:, i:end_i] = F.linear(x_flat, w_chunk_float)
                
            if len(x.shape) == 3:
                y = y.view(x.shape[0], x.shape[1], N)
                
            if self.bias is not None:
                y = y + self.bias
                
            return y

class CommManager:
    """Helper class for managing communication buffers and optimized operations"""
    def __init__(self, hidden_dim, dtype=torch.float32, device='cpu'):
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.device = device
        # Buffer structure: [batch(1), seq(1), dim(1), start_pos(1), task_id(1), timestamp(1), data(...)]
        self.decode_buffer_size = 6 + hidden_dim
        self.decode_buffers = {} # task_id -> buffer
        self.shared_recv_buffer = torch.zeros(self.decode_buffer_size, dtype=self.dtype, device=self.device)
        self.meta_buffer = torch.zeros(6, dtype=torch.float64, device=self.device) # Fixed size metadata buffer
        self.send_reqs = {} # task_id -> req

    def pack_decode(self, x: torch.Tensor, start_pos: int, task_id: int):
        self.ensure_send_complete(task_id)
        if task_id not in self.decode_buffers:
            self.decode_buffers[task_id] = torch.zeros(self.decode_buffer_size, dtype=self.dtype, device=self.device)
        buffer = self.decode_buffers[task_id]
        buffer[0], buffer[1], buffer[2] = 1.0, 1.0, float(self.hidden_dim)
        buffer[3], buffer[4] = float(start_pos), float(task_id)
        buffer[5] = time.perf_counter()
        buffer[6:].copy_(x.flatten())
        return buffer

    def pack_meta(self, shape, start_pos, task_id):
        # shape: (b, s, d)
        self.meta_buffer[0], self.meta_buffer[1], self.meta_buffer[2] = float(shape[0]), float(shape[1]), float(shape[2])
        self.meta_buffer[3], self.meta_buffer[4] = float(start_pos), float(task_id)
        self.meta_buffer[5] = time.perf_counter()
        return self.meta_buffer

    def register_send_req(self, task_id, req):
        self.ensure_send_complete(task_id)
        self.send_reqs[task_id] = req

    def ensure_send_complete(self, task_id):
        if task_id in self.send_reqs and self.send_reqs[task_id] is not None:
            reqs = self.send_reqs[task_id]
            if isinstance(reqs, list):
                for r in reqs: r.wait()
            else:
                reqs.wait()
            self.send_reqs[task_id] = None

    def get_shared_recv_buffer(self):
        return self.shared_recv_buffer
    
    def unpack_shared_buffer(self):
        buffer = self.shared_recv_buffer
        b, s, d = int(buffer[0].item()), int(buffer[1].item()), int(buffer[2].item())
        start_pos, task_id = int(buffer[3].item()), int(buffer[4].item())
        timestamp = buffer[5].item()
        x = buffer[6:].view(b, s, d).clone()
        return x, start_pos, task_id, timestamp
        
    def get_meta_buffer(self):
        return self.meta_buffer
        
    def unpack_meta_buffer(self):
        b, s, d = int(self.meta_buffer[0].item()), int(self.meta_buffer[1].item()), int(self.meta_buffer[2].item())
        start_pos, task_id = int(self.meta_buffer[3].item()), int(self.meta_buffer[4].item())
        timestamp = self.meta_buffer[5].item()
        return (b, s, d), start_pos, task_id, timestamp

def read_m_file_config(model_path):
    if not model_path or not model_path.endswith('.m'): return {}
    try:
        with open(model_path, 'rb') as f:
            header_bytes = f.read(136)
            config_dict = {}
            # Keys: 2: dim, 1: hidden?, 3: n_heads, 4: n_kv, 5: n_layers, 9: vocab
            import struct
            for i in range(8, 136, 8):
                key, val = struct.unpack('<II', header_bytes[i:i+8])
                config_dict[key] = val
            
            cfg = {}
            if 2 in config_dict: cfg['dim'] = config_dict[2]
            
            # Key 3 seems to be hidden_dim (intermediate_size) in this format
            if 3 in config_dict: cfg['hidden_dim'] = config_dict[3]
            
            # Key 1 is suspicious (very large value), ignore it for hidden_dim
            # if 1 in config_dict: cfg['hidden_dim'] = config_dict[1]
            
            # Guess n_heads if not present or if derived head_dim is 0
            # Qwen usually has head_dim = 128
            if 'dim' in cfg:
                cfg['n_heads'] = cfg['dim'] // 128
                cfg['head_dim'] = 128
                
            if 4 in config_dict: cfg['n_kv_heads'] = config_dict[4]
            if 5 in config_dict: cfg['n_layers'] = config_dict[5]
            if 9 in config_dict: cfg['vocab_size'] = config_dict[9]
            
            # Fallback for n_kv_heads
            if 'n_kv_heads' not in cfg and 'n_heads' in cfg:
                 cfg['n_kv_heads'] = cfg['n_heads']
            
            # Sanity check for n_kv_heads
            if 'n_kv_heads' in cfg and 'n_heads' in cfg:
                if cfg['n_kv_heads'] > cfg['n_heads']:
                    print(f"[Config] Warning: n_kv_heads ({cfg['n_kv_heads']}) > n_heads ({cfg['n_heads']}). Clamping to n_heads.")
                    cfg['n_kv_heads'] = cfg['n_heads']

            print(f"[Config] Auto-detected config from .m file: {cfg}")
            return cfg
    except Exception as e:
        print(f"[Config] Failed to read .m config: {e}")
        return {}

# --- Model Components ---

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight

def compute_rope_params(head_dim, theta_base=1000000, context_length=2048):
    freqs = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(context_length)
    freqs = torch.outer(t, freqs)
    # Duplicate for real/imag parts to match x shape
    freqs = torch.cat([freqs, freqs], dim=1)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin

def apply_rope(x, cos, sin, offset=0):
    # x: [b, seq, heads, head_dim]
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2:]
    
    # Cos/Sin: [seq, head_dim/2] -> [1, seq, 1, head_dim/2]
    seq_len = x.shape[1]
    c = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(2)
    s = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(2)
    
    rotated = torch.cat((-x2, x1), dim=-1)
    # x_rotated = x * c + rotated * s
    # Note: Traditional RoPE definition. 
    # Qwen/Llama use complex rotation, which is equivalent to:
    # (x + iy) * (c + is) = (xc - ys) + i(xs + yc)
    # x_out = xc - ys
    # y_out = xs + yc
    # Here: x1 corresponds to real part, x2 to imaginary part? 
    # Usually packed as [x0, x1, x2, x3...] -> pairs (x0, x1).
    # Let's stick to the implementation from static_inference.py for consistency.
    x_rotated = (x * c) + (rotated * s)
    return x_rotated.to(dtype=x.dtype)

class FeedForward(nn.Module):
    def __init__(self, cfg, tp_rank, tp_world_size, tp_group=None, tp_ratios=None, profiler=None):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size
        self.tp_group = tp_group
        self.profiler = profiler
        self.disable_tp = cfg.disable_tp
        
        # Non-uniform splitting logic
        if tp_ratios is None: tp_ratios = [1] * tp_world_size
        total_ratio = sum(tp_ratios)
        start_ratio = sum(tp_ratios[:tp_rank])
        end_ratio = sum(tp_ratios[:tp_rank + 1])
        
        start_idx = int(round(start_ratio * cfg.hidden_dim / total_ratio))
        end_idx = int(round(end_ratio * cfg.hidden_dim / total_ratio))
        self.local_hidden_dim = end_idx - start_idx
        
        # Use QuantizedLinear instead of nn.Linear
        self.gate_proj = QuantizedLinear(cfg.dim, self.local_hidden_dim, dtype=cfg.dtype)
        self.up_proj = QuantizedLinear(cfg.dim, self.local_hidden_dim, dtype=cfg.dtype)
        self.down_proj = QuantizedLinear(self.local_hidden_dim, cfg.dim, dtype=cfg.dtype)

    def forward(self, x):
        # x: [batch, seq, dim]
        # gate/up: [dim, local_hidden] -> [batch, seq, local_hidden]
        x_gate = self.gate_proj(x)
        x_up = self.up_proj(x)
        x_intermediate = F.silu(x_gate) * x_up
        x_out = self.down_proj(x_intermediate)
        
        if self.tp_world_size > 1 and not self.disable_tp:
            t0 = time.perf_counter()
            dist.all_reduce(x_out, op=dist.ReduceOp.SUM, group=self.tp_group)
            if self.profiler: self.profiler.record_comm(time.perf_counter() - t0)
            
        return x_out

class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg, tp_rank, tp_world_size, tp_group=None, tp_ratios=None, profiler=None):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size
        self.tp_group = tp_group
        self.profiler = profiler
        self.disable_tp = cfg.disable_tp
        self.config = cfg
        self.num_heads = cfg.n_heads
        self.num_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        
        if tp_ratios is None: tp_ratios = [1] * tp_world_size
        total_ratio = sum(tp_ratios)
        start_ratio = sum(tp_ratios[:tp_rank])
        end_ratio = sum(tp_ratios[:tp_rank + 1])
        
        # Calculate local heads
        start_head_idx = int(round(start_ratio * self.num_heads / total_ratio))
        end_head_idx = int(round(end_ratio * self.num_heads / total_ratio))
        self.local_num_heads = end_head_idx - start_head_idx
        
        # KV Mapping (GQA)
        # Each query head h maps to kv head h // group_size
        group_size = self.num_heads // self.num_kv_heads
        needed_kv_indices = set()
        self.q_head_to_kv_head = [] 
        
        for h in range(start_head_idx, end_head_idx):
            kv_idx = h // group_size
            needed_kv_indices.add(kv_idx)
            
        self.kv_head_indices = sorted(list(needed_kv_indices))
        self.local_kv_heads = len(self.kv_head_indices)
        
        # Map local Q index -> local KV index
        global_kv_to_local = {k: i for i, k in enumerate(self.kv_head_indices)}
        for h in range(start_head_idx, end_head_idx):
            global_kv = h // group_size
            self.q_head_to_kv_head.append(global_kv_to_local[global_kv])
        self.q_head_to_kv_head = torch.tensor(self.q_head_to_kv_head, dtype=torch.long)
        
        self.d_out = self.local_num_heads * self.head_dim
        self.d_kv = self.local_kv_heads * self.head_dim
        
        # Quantized Projections
        self.q_proj = QuantizedLinear(cfg.dim, self.d_out, bias=True, dtype=cfg.dtype)
        self.k_proj = QuantizedLinear(cfg.dim, self.d_kv, bias=True, dtype=cfg.dtype)
        self.v_proj = QuantizedLinear(cfg.dim, self.d_kv, bias=True, dtype=cfg.dtype)
        self.o_proj = QuantizedLinear(self.d_out, cfg.dim, bias=False, dtype=cfg.dtype)
        
        # KV Cache (per task_id)
        self.k_cache = {}
        self.v_cache = {}

    def forward(self, x, mask, cos, sin, start_pos=0, task_id=0):
        b, seq_len, _ = x.shape
        
        # QKV
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)
        
        # Reshape
        xq = xq.view(b, seq_len, self.local_num_heads, self.head_dim)
        xk = xk.view(b, seq_len, self.local_kv_heads, self.head_dim)
        xv = xv.view(b, seq_len, self.local_kv_heads, self.head_dim)
        
        # RoPE
        # xq = apply_rope(xq, cos, sin, offset=start_pos)
        # xk = apply_rope(xk, cos, sin, offset=start_pos)
        
        # Optimized RoPE: In-place if possible?
        # RoPE is memory bandwidth bound.
        # But we need to use cos/sin slices.
        # Let's just inline apply_rope logic to avoid function overhead?
        
        head_dim = self.head_dim
        
        # Slicing cos/sin for current position
        # Cos/Sin: [max_seq, head_dim/2]
        # We need [seq_len, head_dim/2]
        # Pre-slice to avoid indexing inside the operation
        # Note: cos is shape [max_seq, head_dim].
        # wait, compute_rope_params returns [max_seq, head_dim].
        # In apply_rope, it was:
        # c = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(2)
        # So c is [1, seq, 1, head_dim].
        
        # In my optimized logic:
        # c = cos[start_pos : start_pos + seq_len, :].view(1, seq_len, 1, head_dim // 2)
        # This view assumes cos has head_dim/2 elements?
        # Let's check compute_rope_params:
        # freqs = torch.cat([freqs, freqs], dim=1) -> shape [max_seq, head_dim]
        # So cos is [max_seq, head_dim].
        
        # The traditional RoPE implementation (as in apply_rope):
        # x_rotated = (x * c) + (rotated * s)
        # Here x is [b, seq, heads, head_dim]
        # c should be broadcastable to x.
        # If cos is [seq, head_dim], then c is [1, seq, 1, head_dim].
        
        # My optimized fast_rope logic:
        # x1 = x_in[..., : head_dim // 2]
        # x2 = x_in[..., head_dim // 2 :]
        # rotated = torch.cat((-x2, x1), dim=-1)
        # return (x_in * c) + (rotated * s)
        
        # Here x_in is [b, seq, heads, head_dim].
        # rotated is same shape.
        # So c should be [1, seq, 1, head_dim].
        
        # My broken code:
        # c = cos[...].view(1, seq, 1, head_dim // 2)
        # This tries to shrink it to half? Why?
        # Ah, because some RoPE implementations store complex numbers or pre-sliced cos.
        # But here cos is full head_dim.
        
        c = cos[start_pos : start_pos + seq_len, :].view(1, seq_len, 1, head_dim)
        s = sin[start_pos : start_pos + seq_len, :].view(1, seq_len, 1, head_dim)
        
        def fast_rope(x_in):
            # x_in: [b, seq, heads, head_dim]
            # Standard RoPE:
            # [x0, x1, x2, ...] -> [-x1, x0, -x3, x2, ...]
            # This implementation does:
            # x1 = first half, x2 = second half
            # rotated = [-x2, x1]
            # This matches "half-rotation" style RoPE (common in HF/Llama).
            
            x1 = x_in[..., : head_dim // 2]
            x2 = x_in[..., head_dim // 2 :]
            rotated = torch.cat((-x2, x1), dim=-1)
            return (x_in * c) + (rotated * s)

        xq = fast_rope(xq).to(x.dtype)
        xk = fast_rope(xk).to(x.dtype)
        
        # KV Cache Management
        if task_id not in self.k_cache:
            self.k_cache[task_id] = torch.zeros(b, self.config.max_seq_len, self.local_kv_heads, self.head_dim, device=x.device, dtype=x.dtype)
            self.v_cache[task_id] = torch.zeros(b, self.config.max_seq_len, self.local_kv_heads, self.head_dim, device=x.device, dtype=x.dtype)
            
        # Update cache (Static KV Cache - No cat!)
        # self.k_cache is pre-allocated.
        # Just assign.
        self.k_cache[task_id][:, start_pos : start_pos + seq_len, :, :] = xk
        self.v_cache[task_id][:, start_pos : start_pos + seq_len, :, :] = xv
        
        # Retrieve cached keys/values
        # We only need the valid part for attention
        # View avoids copy?
        keys = self.k_cache[task_id][:, :start_pos + seq_len, :, :]
        values = self.v_cache[task_id][:, :start_pos + seq_len, :, :]
        
        # Expand keys/values for GQA
        if self.local_num_heads > 0:
            # GQA Expansion
            # keys: [b, seq, n_kv, head_dim]
            # indices: [n_local_heads] mapping to n_kv
            
            # Optimization: If n_kv == n_heads (MHA), skip index_select
            if self.num_heads == self.num_kv_heads:
                pass 
            else:
                indices = self.q_head_to_kv_head.to(keys.device)
                keys = keys.index_select(2, indices)
                values = values.index_select(2, indices)
            
        # Attention
        xq = xq.transpose(1, 2) # [b, n_heads, seq, dim]
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Scaled Dot Product Attention
        # scores = torch.matmul(xq, keys.transpose(2, 3)) / (self.head_dim ** 0.5)
        
        # Optimization: Use F.scaled_dot_product_attention if available (PyTorch 2.0+)
        # It handles masking and scaling efficiently.
        # But we need to handle causal mask manually if not using is_causal=True
        # For inference (start_pos > 0), we query 1 token against N keys. It's not strictly "causal" triangle mask
        # It's [1, N] attention.
        
        if hasattr(F, 'scaled_dot_product_attention') and xq.dtype != torch.int8:
             # F.sdpa expects [b, n_heads, seq_q, head_dim]
             # keys: [b, n_heads, seq_k, head_dim]
             # If start_pos > 0 (decoding), we don't need mask usually? 
             # Wait, all previous keys are valid.
             # If seq_len > 1 (prefill), we need causal mask.
             
             is_causal = (seq_len > 1) and (mask is None) # If mask provided, use it
             # Actually for decoding (seq=1), is_causal=False is fine.
             
             # Note: F.sdpa might not support some dtypes on CPU or specific hardware.
             # Let's try it.
             output = F.scaled_dot_product_attention(xq, keys, values, attn_mask=mask, dropout_p=0.0, is_causal=is_causal)
             
        else:
            scores = torch.matmul(xq, keys.transpose(2, 3)) / (self.head_dim ** 0.5)
            if mask is not None:
                scores = scores.masked_fill(mask, float("-inf"))
            probs = F.softmax(scores, dim=-1)
            output = torch.matmul(probs, values)
        
        output = output.transpose(1, 2).contiguous().view(b, seq_len, -1)
        out = self.o_proj(output)
        
        if self.tp_world_size > 1 and not self.disable_tp:
            t0 = time.perf_counter()
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
            if self.profiler: self.profiler.record_comm(time.perf_counter() - t0)
            
        return out

class TransformerBlock(nn.Module):
    def __init__(self, cfg, tp_rank, tp_world_size, tp_group=None, tp_ratios=None, profiler=None):
        super().__init__()
        self.attention = GroupedQueryAttention(cfg, tp_rank, tp_world_size, tp_group, tp_ratios, profiler)
        self.feed_forward = FeedForward(cfg, tp_rank, tp_world_size, tp_group, tp_ratios, profiler)
        self.attention_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)

    def forward(self, x, mask, cos, sin, start_pos=0, task_id=0):
        h = x + self.attention(self.attention_norm(x), mask, cos, sin, start_pos, task_id)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class OptimizedDistributedQwen3Model(nn.Module):
    def __init__(self, config, my_config, tp_group=None):
        super().__init__()
        self.config = config
        self.tp_group = tp_group
        self.profiler = InferenceProfiler()
        self.my_rank = my_config['my_rank']
        self.world_size = my_config['world_size']
        
        self.comm_manager = CommManager(config.dim, dtype=config.dtype)
        self.my_stage_idx = my_config['my_stage_idx']
        self.stage_ranks = my_config['stage_ranks']
        self.layers_per_stage = my_config['layers_per_stage']
        
        self.tp_group_ranks = self.stage_ranks[self.my_stage_idx]
        self.tp_world_size = len(self.tp_group_ranks)
        self.tp_rank = self.tp_group_ranks.index(self.my_rank)
        
        self.tp_ratios = my_config.get('tp_ratios', [1] * self.tp_world_size)
        
        self.start_layer = sum(self.layers_per_stage[:self.my_stage_idx])
        self.num_layers = self.layers_per_stage[self.my_stage_idx]
        
        logger.info(f"[Rank {self.my_rank}] Init Optimized Model: Stage {self.my_stage_idx}, TP {self.tp_rank}/{self.tp_world_size}")
        
        self.layers = nn.ModuleList([
            TransformerBlock(config, self.tp_rank, self.tp_world_size, self.tp_group, self.tp_ratios, self.profiler) 
            for _ in range(self.num_layers)
        ])
        
        cos, sin = compute_rope_params(config.head_dim, theta_base=config.rope_theta, context_length=config.max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
        if self.my_stage_idx == 0:
            self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        if self.my_stage_idx == len(self.stage_ranks) - 1:
            self.norm = RMSNorm(config.dim, config.norm_eps)
            # Use QuantizedLinear for lm_head to save memory (Int8 weights)
            self.output = QuantizedLinear(config.dim, config.vocab_size, bias=False, dtype=config.dtype)

    def load_weights(self, model_path):
        logger.info(f"[Rank {self.my_rank}] Loading weights from {model_path}...")
        
        if model_path.endswith('.m'):
             self._load_m_file(model_path)
        else:
             self._load_safetensors(model_path)
             
    def _load_m_file(self, model_path):
        # 1. Attempt to load as safetensors (handling renamed files)
        try:
            # Check header first to avoid long timeout or weird error
            with open(model_path, 'rb') as f:
                header = f.read(8)
                # Safetensors usually starts with a large int (json length)
                # If it's the custom binary, it starts with magic or config?
                # optimized_inference.py says header parsing starts at offset 8.
                pass
                
            logger.info(f"[Rank {self.my_rank}] Attempting to load .m file as safetensors...")
            self._load_safetensors(model_path)
            logger.info(f"[Rank {self.my_rank}] Successfully loaded .m file as safetensors.")
            return
        except Exception as e:
            logger.info(f"[Rank {self.my_rank}] Not a safetensors file: {e}")

        # 2. Attempt to load as Custom Binary (.m)
        try:
            logger.info(f"[Rank {self.my_rank}] Attempting to load .m file as Custom Binary (Q8_0)...")
            self._load_m_file_binary(model_path)
            logger.info(f"[Rank {self.my_rank}] Successfully loaded .m file as Custom Binary.")
            return
        except Exception as e:
            logger.warning(f"[Rank {self.my_rank}] Failed to load as Custom Binary: {e}")

        # 3. Fallback to Random
        logger.warning(f"[Rank {self.my_rank}] Falling back to heuristic/random initialization (OUTPUT WILL BE GARBLED).")
        
        def init_quantized(layer):
            layer.weight_data = torch.randint(-127, 127, (layer.out_features, layer.in_features), dtype=torch.int8)
            layer.scale = torch.ones((layer.out_features, 1), dtype=torch.float32) * (1.0/127.0)
            # layer.float_weight = layer.weight_data.to(torch.float32) * layer.scale # Removed float_weight
            
        for layer in self.layers:
            init_quantized(layer.attention.q_proj)
            init_quantized(layer.attention.k_proj)
            init_quantized(layer.attention.v_proj)
            init_quantized(layer.attention.o_proj)
            init_quantized(layer.feed_forward.gate_proj)
            init_quantized(layer.feed_forward.up_proj)
            init_quantized(layer.feed_forward.down_proj)

    def _load_m_file_binary(self, model_path):
        import mmap
        import struct
        
        with open(model_path, 'rb') as f:
            # Parse Header
            header_bytes = f.read(136)
            config_dict = {}
            for i in range(8, 136, 8):
                key, val = struct.unpack('<II', header_bytes[i:i+8])
                config_dict[key] = val
            
            # Basic validation
            file_dim = config_dict.get(2, self.config.dim)
            file_vocab_size = config_dict.get(9, self.config.vocab_size)
            
            if file_dim != self.config.dim:
                logger.warning(f"Binary file dim {file_dim} mismatch with config {self.config.dim}")
            
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            offset = 136
            
            # Helper for reading tensors
            def load_tensor(n_bytes, dtype=torch.float32, shape=None):
                nonlocal offset
                data = torch.frombuffer(mm[offset : offset + n_bytes], dtype=dtype)
                if shape:
                    data = data.view(shape)
                offset += n_bytes
                return data.clone() # Clone to copy to memory and detach from mmap

            # 1. Embeddings (Float32)
            # Stage 0 needs embeddings. Others might skip or load dummy to advance offset?
            # We must advance offset regardless of rank to stay in sync with file layout.
            vocab = file_vocab_size
            dim = file_dim
            emb_bytes = vocab * dim * 4
            
            if self.my_stage_idx == 0:
                logger.info(f"[Rank {self.my_rank}] Loading Embeddings (Vocab={vocab}, Dim={dim})...")
                w = load_tensor(emb_bytes, torch.float32, (vocab, dim))
                # Resize if needed
                if vocab != self.config.vocab_size:
                    logger.warning(f"Resizing embeddings from {vocab} to {self.config.vocab_size}")
                    self.tok_embeddings = nn.Embedding(self.config.vocab_size, self.config.dim)
                    # Copy common part
                    min_vocab = min(vocab, self.config.vocab_size)
                    self.tok_embeddings.weight.data[:min_vocab].copy_(w[:min_vocab])
                else:
                    self.tok_embeddings.weight.data.copy_(w)
            else:
                offset += emb_bytes # Skip

            # 2. Layers
            # Assumed Order: RMS_Attn, RMS_FFN, Q, K, V, O, Gate, Up, Down
            # Note: Weights are likely (out, in) flat.
            
            head_dim = self.config.head_dim
            n_heads = self.config.n_heads
            n_kv = self.config.n_kv_heads
            hidden = self.config.hidden_dim
            
            # Sizes (bytes) for Q8_0 (1 byte per element)
            sz_rms = dim * 4
            sz_q = dim * n_heads * head_dim
            sz_k = dim * n_kv * head_dim
            sz_v = dim * n_kv * head_dim
            sz_o = n_heads * head_dim * dim
            sz_gate = dim * hidden
            sz_up = dim * hidden
            sz_down = hidden * dim
            
            # TP Helper
            def load_split_q4(target_layer, full_shape, split_dim, sz_params):
                nonlocal offset
                # Q4_0: Block size 32. Per block: 2 byte scale (f16) + 16 bytes data (32 4-bit).
                # Total 18 bytes per 32 params.
                block_size = 32
                block_bytes = 18
                
                num_blocks = sz_params // block_size
                total_bytes = num_blocks * block_bytes
                
                # Check remaining
                remaining = mm.size() - offset
                if remaining < total_bytes:
                    logger.warning(f"[Rank {self.my_rank}] Short read for Q4! Needed {total_bytes}, got {remaining}. Padding.")
                    # Fallback to zeros if corrupted
                    w_float = torch.zeros(full_shape, dtype=torch.float32)
                    offset += remaining
                else:
                    # Read raw bytes
                    raw_bytes = torch.frombuffer(mm[offset : offset + total_bytes], dtype=torch.uint8).clone()
                    offset += total_bytes
                    
                    # Unpack Q4_0
                    # This is slow in pure Python/Torch without custom kernel, but needed for correctness.
                    # Vectorized unpacking:
                    # Reshape to [num_blocks, 18]
                    blocks = raw_bytes.view(num_blocks, block_bytes)
                    
                    # Scales: First 2 bytes (float16)
                    # scales_bytes is a Tensor of uint8. We need to bitcast to float16.
                    scales_bytes = blocks[:, :2].flatten() # [num_blocks * 2] (contiguous uint8)
                    # Bitcast to float16
                    scales = scales_bytes.view(torch.float16).to(torch.float32)
                    
                    # Data: Next 16 bytes
                    data_bytes = blocks[:, 2:].flatten() # [num_blocks * 16]
                    
                    # Unpack nibbles
                    # w0 = byte & 0x0F
                    # w1 = byte >> 4
                    low = (data_bytes & 0x0F).to(torch.float32) - 8.0
                    high = (data_bytes >> 4).to(torch.float32) - 8.0
                    
                    # Interleave? Usually low is even index, high is odd?
                    # Q4_0 layout: [w0, w1, w2... w31]
                    # Byte 0: w0, w1
                    # So we need to stack them.
                    weights = torch.stack([low, high], dim=1).flatten() # [num_blocks * 32]
                    
                    # Apply scales
                    # scales: [num_blocks] -> repeat 32 times
                    scales_expanded = scales.repeat_interleave(32)
                    
                    w_float = (weights * scales_expanded).view(full_shape)

                # Now we have Float weights. Quantize to Int8 for our runtime
                # Calculate TP split
                total_ratio = sum(self.tp_ratios)
                dim_size = full_shape[split_dim]
                size_per_ratio = dim_size / total_ratio
                
                start = int(round(sum(self.tp_ratios[:self.tp_rank]) * size_per_ratio))
                end = int(round(sum(self.tp_ratios[:self.tp_rank + 1]) * size_per_ratio))
                
                # Slice
                if split_dim == 0:
                    w_slice = w_float[start:end]
                else:
                    w_slice = w_float[:, start:end]
                
                # Quantize to Int8 (Per-Channel)
                abs_w = w_slice.abs()
                max_val = abs_w.max(dim=1, keepdim=True).values
                scale = max_val / 127.0
                scale = torch.where(scale == 0, torch.tensor(1.0, dtype=scale.dtype, device=scale.device), scale)
                # Fix NaNs in scale (from misalignment or bad data)
                if torch.isnan(scale).any():
                    scale = torch.nan_to_num(scale, nan=0.001) # Use small default scale
                    
                w_int8 = (w_slice / scale).round().clamp(-127, 127).to(torch.int8)
                
                target_layer.weight_data = w_int8
                target_layer.scale = scale
                target_layer.out_features = w_int8.shape[0]
                target_layer.in_features = w_int8.shape[1]

            def load_split_q8(target_layer, full_shape, split_dim, sz_bytes):
                # Legacy / Fallback for Q8
                # ... (Keep existing if needed, but we replace usage with load_split_q4)
                pass


            def load_norm(target_norm):
                nonlocal offset
                # Align to 32 bytes before reading F32 Norm?
                # offset = (offset + 31) // 32 * 32 
                w = load_tensor(sz_rms, torch.float32, (dim,))
                target_norm.weight.data.copy_(w)

            def skip_bytes(n):
                nonlocal offset
                offset += n

            def load_bias(target_layer, dim_size):
                nonlocal offset
                sz_bias = dim_size * 4 # Float32
                
                # Check remaining
                if mm.size() - offset < sz_bias:
                    logger.warning(f"Short read for bias. Padding.")
                    offset = mm.size()
                    return
                    
                w = load_tensor(sz_bias, torch.float32)
                
                # TP Split for Bias
                # Bias is 1D [out]. Split along dim 0.
                total_ratio = sum(self.tp_ratios)
                size_per_ratio = dim_size // total_ratio
                start = int(round(sum(self.tp_ratios[:self.tp_rank]) * size_per_ratio))
                end = int(round(sum(self.tp_ratios[:self.tp_rank + 1]) * size_per_ratio))
                
                if target_layer.bias is not None:
                    target_layer.bias.data.copy_(w[start:end])
                
            # Iterate over all layers in the file
            total_layers_in_file = self.config.n_layers # Use config count
            
            for i in range(total_layers_in_file):
                # Check if this layer belongs to my stage
                is_my_layer = (i >= self.start_layer) and (i < self.start_layer + self.num_layers)
                
                if is_my_layer:
                    local_idx = i - self.start_layer
                    block = self.layers[local_idx]
                    
                    # Q, K, V, O
                    # Q: (dim, dim)
                    load_split_q4(block.attention.q_proj, (n_heads*head_dim, dim), 0, sz_q // 1 * 1)
                    # load_bias(block.attention.q_proj, n_heads*head_dim)
                    
                    # K: (dim_kv, dim)
                    load_split_q4(block.attention.k_proj, (n_kv*head_dim, dim), 0, sz_k)
                    # load_bias(block.attention.k_proj, n_kv*head_dim)
                    
                    # V: (dim_kv, dim)
                    load_split_q4(block.attention.v_proj, (n_kv*head_dim, dim), 0, sz_v)
                    # load_bias(block.attention.v_proj, n_kv*head_dim)
                    
                    # O: (dim, dim)
                    load_split_q4(block.attention.o_proj, (dim, n_heads*head_dim), 1, sz_o)
                    # O usually no bias
                    
                    # RMS Attn (Moved to after O to fix alignment)
                    load_norm(block.attention_norm)
                    
                    # RMS FFN (Post Attention Norm)
                    load_norm(block.ffn_norm)

                    # Gate, Up, Down
                    load_split_q4(block.feed_forward.gate_proj, (hidden, dim), 0, sz_gate)
                    load_split_q4(block.feed_forward.up_proj, (hidden, dim), 0, sz_up)
                    load_split_q4(block.feed_forward.down_proj, (dim, hidden), 1, sz_down)
                    
                else:
                    # Skip this layer
                    # Q4 params
                    total_params = (n_heads*head_dim*dim) + (n_kv*head_dim*dim)*2 + (dim*n_heads*head_dim) + (hidden*dim)*2 + (dim*hidden)
                    skip_size = (total_params // 32) * 18 + sz_rms * 2
                    # Add biases skip
                    # bias_size = (n_heads*head_dim + n_kv*head_dim*2) * 4
                    skip_bytes(skip_size) # + bias_size)

            # 3. Final Norm & Head
            # Last stage loads
            is_last_stage = (self.my_stage_idx == len(self.stage_ranks) - 1)
            
            # Final Norm
            if is_last_stage:
                load_norm(self.norm)
            else:
                skip_bytes(sz_rms)
                
            # Head (if present)
            # Check remaining bytes
            remaining = mm.size() - offset
            sz_head_float = vocab * dim * 4
            sz_head_q4 = (vocab * dim // 32) * 18
            
            if is_last_stage:
                if remaining >= sz_head_float:
                    logger.info(f"[Rank {self.my_rank}] Loading Head (Float32) as Int8...")
                    # Process in chunks to save memory
                    chunk_rows = 1024
                    
                    # Prepare Int8 storage first
                    w_int8_full = torch.empty((vocab, dim), dtype=torch.int8)
                    scale_full = torch.empty((vocab, 1), dtype=torch.float32)
                    
                    for i in range(0, vocab, chunk_rows):
                        end_i = min(i + chunk_rows, vocab)
                        rows = end_i - i
                        n_bytes = rows * dim * 4
                        
                        # Read chunk (Float32)
                        w_chunk = load_tensor(n_bytes, torch.float32, (rows, dim))
                        
                        # Quantize chunk
                        abs_w = w_chunk.abs()
                        max_val = abs_w.max(dim=1, keepdim=True).values
                        scale = max_val / 127.0
                        scale = torch.where(scale == 0, torch.tensor(1.0, dtype=scale.dtype, device=scale.device), scale)
                        w_int8 = (w_chunk / scale).round().clamp(-127, 127).to(torch.int8)
                        
                        w_int8_full[i:end_i] = w_int8
                        scale_full[i:end_i] = scale
                        
                        del w_chunk, w_int8, scale
                        gc.collect()

                    self.output.weight_data = w_int8_full
                    self.output.scale = scale_full
                    self.output.out_features = vocab
                    self.output.in_features = dim

                elif remaining >= sz_head_q4:
                    logger.info(f"[Rank {self.my_rank}] Loading Head (Q4_0) as Int8 (Chunked)...")
                    
                    # Prepare Int8 storage
                    w_int8_full = torch.empty((vocab, dim), dtype=torch.int8)
                    scale_full = torch.empty((vocab, 1), dtype=torch.float32)
                    
                    # Chunk processing
                    chunk_rows = 1024
                    block_size = 32
                    block_bytes = 18
                    
                    for i in range(0, vocab, chunk_rows):
                        end_i = min(i + chunk_rows, vocab)
                        rows = end_i - i
                        num_blocks = (rows * dim) // block_size
                        chunk_bytes = num_blocks * block_bytes
                        
                        # Read raw bytes for chunk
                        raw_bytes = torch.frombuffer(mm[offset : offset + chunk_bytes], dtype=torch.uint8)
                        offset += chunk_bytes
                        
                        # Unpack Q4
                        blocks = raw_bytes.view(num_blocks, block_bytes)
                        scales_bytes = blocks[:, :2].flatten()
                        scales = scales_bytes.view(torch.float16).to(torch.float32)
                        
                        data_bytes = blocks[:, 2:].flatten()
                        low = (data_bytes & 0x0F).to(torch.float32) - 8.0
                        high = (data_bytes >> 4).to(torch.float32) - 8.0
                        weights = torch.stack([low, high], dim=1).flatten()
                        
                        scales_expanded = scales.repeat_interleave(32)
                        w_float_chunk = (weights * scales_expanded).view(rows, dim)
                        
                        # Quantize to Int8
                        abs_w = w_float_chunk.abs()
                        max_val = abs_w.max(dim=1, keepdim=True).values
                        scale = max_val / 127.0
                        scale = torch.where(scale == 0, torch.tensor(1.0, dtype=scale.dtype, device=scale.device), scale)
                        if torch.isnan(scale).any(): scale = torch.nan_to_num(scale, nan=0.001)

                        w_int8 = (w_float_chunk / scale).round().clamp(-127, 127).to(torch.int8)
                        
                        w_int8_full[i:end_i] = w_int8
                        scale_full[i:end_i] = scale
                        
                        del raw_bytes, blocks, scales, data_bytes, weights, scales_expanded, w_float_chunk, w_int8
                        # gc.collect() # Frequent GC might slow down loop, maybe every few iterations?
                    
                    self.output.weight_data = w_int8_full
                    self.output.scale = scale_full
                    self.output.out_features = vocab
                    self.output.in_features = dim
                    
                else:
                    logger.warning(f"[Rank {self.my_rank}] Head weights missing (Remaining {remaining}, Needed {sz_head_q4}). Using random.")
                    # Random init for QuantizedLinear
                    self.output.weight_data = torch.randint(-127, 127, (vocab, dim), dtype=torch.int8)
                    self.output.scale = torch.ones((vocab, 1), dtype=torch.float32) * 0.01
            else:
                if remaining >= sz_head_float:
                    skip_bytes(sz_head_float)
                elif remaining >= sz_head_q4:
                    skip_bytes(sz_head_q4)

    def _load_safetensors(self, model_path):
        logger.info(f"[Rank {self.my_rank}] Loading weights from {model_path} (SafeTensors)...")
        
        def get_tensor(key):
            with safe_open(model_path, framework="pt", device="cpu") as f:
                return f.get_tensor(key)
        
        def load_split(target_layer, key, dim):
            # target_layer is QuantizedLinear
            # We load directly into weight_data (quantizing from float if needed)
            w = get_tensor(key).to(torch.float32)
            
            total_ratio = sum(self.tp_ratios)
            dim_size = w.shape[dim]
            size_per_ratio = dim_size / total_ratio
            
            start = int(round(sum(self.tp_ratios[:self.tp_rank]) * size_per_ratio))
            end = int(round(sum(self.tp_ratios[:self.tp_rank + 1]) * size_per_ratio))
            
            if dim == 0:
                w_slice = w[start:end]
            else:
                w_slice = w[:, start:end]
            
            # Quantize weights to Int8
            # Per-Channel Quantization (dim 0 of w_slice)
            abs_w = w_slice.abs()
            # max per row
            max_val = abs_w.max(dim=1, keepdim=True).values
            scale = max_val / 127.0
            # Avoid divide by zero
            scale = torch.where(scale == 0, torch.tensor(1.0, dtype=scale.dtype, device=scale.device), scale)
                
            w_int8 = (w_slice / scale).round().clamp(-127, 127).to(torch.int8)
            
            target_layer.weight_data = w_int8
            target_layer.scale = scale
            target_layer.out_features = w_int8.shape[0]
            target_layer.in_features = w_int8.shape[1]
            
            del w
            gc.collect()

        def load_split_heads(target_layer, key, dim, num_groups, head_dim, indices=None):
            w = get_tensor(key).to(torch.float32)
            total_ratio = sum(self.tp_ratios)
            
            slice_data = None
            
            if indices is None:
                start_ratio = sum(self.tp_ratios[:self.tp_rank])
                end_ratio = sum(self.tp_ratios[:self.tp_rank + 1])
                
                start_group = int(round(start_ratio * num_groups / total_ratio))
                end_group = int(round(end_ratio * num_groups / total_ratio))
                
                start_idx = start_group * head_dim
                end_idx = end_group * head_dim
                
                if start_idx == end_idx:
                    slice_data = torch.empty(0, 0) # Should handle empty?
                else:
                    if dim == 0:
                        slice_data = w[start_idx:end_idx]
                    else:
                        slice_data = w[:, start_idx:end_idx]
            else:
                if len(indices) == 0:
                    slice_data = torch.empty(0, 0)
                else:
                    slices = []
                    for idx in indices:
                        start = idx * head_dim
                        end = (idx + 1) * head_dim
                        if dim == 0:
                            slices.append(w[start:end])
                        else:
                            slices.append(w[:, start:end])
                    
                    if dim == 0:
                        slice_data = torch.cat(slices, dim=0)
                    else:
                        slice_data = torch.cat(slices, dim=1)
            
            if slice_data is not None:
                # Quantize Per-Channel
                abs_w = slice_data.abs()
                max_val = abs_w.max(dim=1, keepdim=True).values
                scale = max_val / 127.0
                scale = torch.where(scale == 0, torch.tensor(1.0, dtype=scale.dtype, device=scale.device), scale)
                
                w_int8 = (slice_data / scale).round().clamp(-127, 127).to(torch.int8)
                
                target_layer.weight_data = w_int8
                target_layer.scale = scale
                target_layer.out_features = w_int8.shape[0]
                target_layer.in_features = w_int8.shape[1]

            del w
            gc.collect()

        def load_bias_split_heads(target_layer, key, num_groups, head_dim, indices=None):
            # Helper for bias loading (always dim 0 since bias is 1D)
            if target_layer.bias is None: return
            
            # Check if key exists
            with safe_open(model_path, framework="pt", device="cpu") as f:
                if key not in f.keys(): return
                
            w = get_tensor(key).to(torch.float32)
            # Logic similar to split_heads but 1D
            
            slice_data = None
            
            if indices is None:
                total_ratio = sum(self.tp_ratios)
                start_ratio = sum(self.tp_ratios[:self.tp_rank])
                end_ratio = sum(self.tp_ratios[:self.tp_rank + 1])
                
                start_group = int(round(start_ratio * num_groups / total_ratio))
                end_group = int(round(end_ratio * num_groups / total_ratio))
                
                start_idx = start_group * head_dim
                end_idx = end_group * head_dim
                
                if start_idx != end_idx:
                    slice_data = w[start_idx:end_idx]
            else:
                if len(indices) > 0:
                    slices = []
                    for idx in indices:
                        start = idx * head_dim
                        end = (idx + 1) * head_dim
                        slices.append(w[start:end])
                    slice_data = torch.cat(slices, dim=0)
            
            if slice_data is not None:
                target_layer.bias.data.copy_(slice_data)
                
            del w
            gc.collect()

        def load_layer_weights(layer_idx, block):
            prefix = f"model.layers.{layer_idx}."
            
            # Attention
            # Q, K, V are row parallel (split dim 0)
            load_split_heads(block.attention.q_proj, f"{prefix}self_attn.q_proj.weight", 0, self.config.n_heads, self.config.head_dim)
            load_bias_split_heads(block.attention.q_proj, f"{prefix}self_attn.q_proj.bias", self.config.n_heads, self.config.head_dim)
            
            load_split_heads(block.attention.k_proj, f"{prefix}self_attn.k_proj.weight", 0, self.config.n_kv_heads, self.config.head_dim, indices=block.attention.kv_head_indices)
            load_bias_split_heads(block.attention.k_proj, f"{prefix}self_attn.k_proj.bias", self.config.n_kv_heads, self.config.head_dim, indices=block.attention.kv_head_indices)
            
            load_split_heads(block.attention.v_proj, f"{prefix}self_attn.v_proj.weight", 0, self.config.n_kv_heads, self.config.head_dim, indices=block.attention.kv_head_indices)
            load_bias_split_heads(block.attention.v_proj, f"{prefix}self_attn.v_proj.bias", self.config.n_kv_heads, self.config.head_dim, indices=block.attention.kv_head_indices)
            
            # O is col parallel (split dim 1)
            load_split_heads(block.attention.o_proj, f"{prefix}self_attn.o_proj.weight", 1, self.config.n_heads, self.config.head_dim)
            # O bias? Usually no bias in O, but if there is, it's not split (reduce sum output)
            # If there is bias, it is applied after all reduce?
            # Standard Llama/Qwen doesn't have O bias.
            
            # FFN
            # Gate, Up are row parallel (split dim 0)
            load_split(block.feed_forward.gate_proj, f"{prefix}mlp.gate_proj.weight", 0)
            load_split(block.feed_forward.up_proj, f"{prefix}mlp.up_proj.weight", 0)
            # Down is col parallel (split dim 1)
            load_split(block.feed_forward.down_proj, f"{prefix}mlp.down_proj.weight", 1)
            
            # Norms (Replicated)
            w = get_tensor(f"{prefix}input_layernorm.weight").to(torch.float32)
            block.attention_norm.weight.data.copy_(w)
            
            w = get_tensor(f"{prefix}post_attention_layernorm.weight").to(torch.float32)
            block.ffn_norm.weight.data.copy_(w)
            
            gc.collect()

        # Load Embeddings (Stage 0)
        if self.my_stage_idx == 0:
            w = get_tensor("model.embed_tokens.weight").to(torch.float32)
            self.tok_embeddings.weight.data.copy_(w)
            del w
            gc.collect()
            
        # Load Layers
        for i, layer in enumerate(self.layers):
            load_layer_weights(self.start_layer + i, layer)
            
        # Load Final Norm & Head (Last Stage)
        if self.my_stage_idx == len(self.stage_ranks) - 1:
            w = get_tensor("model.norm.weight").to(torch.float32)
            self.norm.weight.data.copy_(w)
            
            with safe_open(model_path, framework="pt", device="cpu") as f:
                if "lm_head.weight" in f.keys():
                    w = f.get_tensor("lm_head.weight")
                elif "model.embed_tokens.weight" in f.keys():
                    w = f.get_tensor("model.embed_tokens.weight")
                else:
                    raise KeyError("lm_head.weight not found")
            
            self.output.weight.data.copy_(w.to(torch.float32))
            
        logger.info(f"[Rank {self.my_rank}] Weights loaded successfully.")

    def forward(self, x, start_pos=0, task_id=0):
        # Stage > 0: Receive
        if self.my_stage_idx > 0:
            prev_root = self.stage_ranks[self.my_stage_idx - 1][0]
            if self.tp_rank == 0:
                t0 = time.perf_counter()
                if start_pos > 0:
                    # Optimized Receive (Decode Phase): Single Packed Buffer
                    recv_buf = self.comm_manager.get_shared_recv_buffer()
                    dist.recv(recv_buf, src=prev_root)
                    x, start_pos, task_id, timestamp = self.comm_manager.unpack_shared_buffer()
                else:
                    # Robust Receive (Prompt Phase): Meta + Data
                    meta_buf = self.comm_manager.get_meta_buffer()
                    dist.recv(meta_buf, src=prev_root)
                    (b, s, d), start_pos, task_id, timestamp = self.comm_manager.unpack_meta_buffer()
                    x = torch.zeros((b, s, d), dtype=self.config.dtype)
                    dist.recv(x, src=prev_root)
                self.profiler.record_comm(time.perf_counter() - t0)
            else:
                x = None
            
            # Broadcast to TP group
            if self.tp_world_size > 1:
                t0 = time.perf_counter()
                root = self.stage_ranks[self.my_stage_idx][0]
                
                if start_pos > 0:
                    # Decode Phase: Broadcast Packed Buffer
                    # Note: If we are rank 0, we already have data in shared_recv_buffer or packed_buffer?
                    # We just unpacked it. We should repack or use the buffer we received.
                    # The shared_recv_buffer in comm_manager holds the data for Rank 0.
                    recv_buf = self.comm_manager.get_shared_recv_buffer()
                    dist.broadcast(recv_buf, src=root, group=self.tp_group)
                    if self.tp_rank != 0:
                        x, start_pos, task_id, timestamp = self.comm_manager.unpack_shared_buffer()
                else:
                    # Prompt Phase: Broadcast Meta + Data
                    meta_buf = self.comm_manager.get_meta_buffer()
                    if self.tp_rank == 0:
                         # We already have meta_buf populated from receive
                         pass 
                    dist.broadcast(meta_buf, src=root, group=self.tp_group)
                    
                    if self.tp_rank != 0:
                        (b, s, d), start_pos, task_id, timestamp = self.comm_manager.unpack_meta_buffer()
                        x = torch.zeros((b, s, d), dtype=self.config.dtype)
                    
                    dist.broadcast(x, src=root, group=self.tp_group)

                self.profiler.record_comm(time.perf_counter() - t0)

        # Stage 0: Embeddings
        if self.my_stage_idx == 0:
            if self.tp_world_size > 1:
                t0 = time.perf_counter()
                root = self.stage_ranks[self.my_stage_idx][0]
                
                if self.tp_rank == 0:
                    shape = torch.tensor(x.shape, dtype=torch.long)
                else:
                    shape = torch.zeros(2, dtype=torch.long)
                
                # Broadcast shape
                dist.broadcast(shape, src=root, group=self.tp_group)
                
                # Prepare buffer
                if self.tp_rank != 0:
                    x = torch.zeros(tuple(shape.tolist()), dtype=torch.long)
                
                # Broadcast data
                dist.broadcast(x, src=root, group=self.tp_group)
                self.profiler.record_comm(time.perf_counter() - t0)

            t0 = time.perf_counter()
            x = self.tok_embeddings(x)
            self.profiler.record_compute(time.perf_counter() - t0)

        # Layers
        b, seq, _ = x.shape
        mask = None
        if seq > 1:
             mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1)
             
        for i, layer in enumerate(self.layers):
            self.profiler.start_layer(self.start_layer + i)
            x = layer(x, mask, self.cos, self.sin, start_pos=start_pos, task_id=task_id)
            self.profiler.end_layer()

        # Send / Output
        if self.my_stage_idx < len(self.stage_ranks) - 1:
            if self.tp_rank == 0:
                t0 = time.perf_counter()
                next_root = self.stage_ranks[self.my_stage_idx + 1][0]
                
                if start_pos > 0 and x.shape[1] == 1:
                    # Optimized Send (Decode Phase): Single Packed Buffer
                    packed_buf = self.comm_manager.pack_decode(x, start_pos, task_id)
                    req = dist.isend(packed_buf, dst=next_root)
                    self.comm_manager.register_send_req(task_id, req)
                else:
                    # Robust Send (Prompt Phase): Meta + Data -> Now Async
                    meta_buf = self.comm_manager.pack_meta(x.shape, start_pos, task_id)
                    req_meta = dist.isend(meta_buf, dst=next_root)
                    
                    # Ensure x is contiguous and ready for sending
                    if not x.is_contiguous(): x = x.contiguous()
                    # We need to keep a reference to x until send is complete, 
                    # but since we are in a loop and x is overwritten or discarded, 
                    # we should register it in comm_manager or rely on Python's ref counting if we block later.
                    # For safety in async pipeline, we register the request.
                    req_data = dist.isend(x, dst=next_root)
                    
                    self.comm_manager.register_send_req(task_id, [req_meta, req_data])
                    
                self.profiler.record_comm(time.perf_counter() - t0)
            return None
        else:
            t0 = time.perf_counter()
            x = self.norm(x)
            if self.tp_rank == 0:
                logits = self.output(x)
                self.profiler.record_compute(time.perf_counter() - t0)
                return logits
            return None

# --- Main Logic (Copied/Adapted from static_inference.py) ---

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
            "devices": stage_devices, "tp_ratios": ratios, "num_layers": num_layers, "root_ip": stage_devices[0]
        })
    return {"stages": parsed_stages, "global_root_ip": parsed_stages[0]["devices"][0]}

def send_config_to_node(ip, port, config):
    if ':' in ip: ip, port = ip.split(':')[0], int(ip.split(':')[1])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(30)
        s.connect((ip, port))
        s.sendall(json.dumps(config).encode('utf-8'))

def start_listener(ip, port):
    if ':' in ip: ip, port = ip.split(':')[0], int(ip.split(':')[1])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((ip, port))
        s.listen(1)
        conn, addr = s.accept()
        with conn:
            data = conn.recv(1024 * 1024)
            return json.loads(data.decode('utf-8'))

class TaskState:
    def __init__(self, task_id, prompt, tokenizer=None, input_ids=None):
        self.task_id = task_id
        self.prompt = prompt
        self.input_ids = input_ids if input_ids is not None else (tokenizer.encode(prompt, return_tensors="pt").to(torch.long) if tokenizer else torch.empty(1, 1, dtype=torch.long))
        self.curr_input_ids = self.input_ids
        self.generated_ids = self.input_ids[0].tolist() if self.input_ids.numel() > 0 else []
        self.start_pos = 0
        self.is_complete = False
        self.token_count = 0
        self.start_time = 0.0
        self.end_time = 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["listen", "assign"])
    parser.add_argument("--my_ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument("--config", type=str)
    parser.add_argument("--ips", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--num_tasks", type=int, default=1, help="Number of concurrent tasks (<= 5)")
    parser.add_argument("--disable_tp", action="store_true", help="Disable TP all_reduce for CPU performance testing")
    parser.add_argument("--test_local", action="store_true")
    parser.add_argument("--test_4", action="store_true")
    args = parser.parse_args()

    if args.test_local:
        test_local_2_devices(args)
        return
    if args.test_4:
        test_local_4_devices()
        return

    my_config = None
    if args.mode == "listen":
        my_config = start_listener(args.my_ip, args.port)
    elif args.mode == "assign":
        ip_list = args.ips.split(',')
        alloc = parse_allocation_config(args.config, ip_list)
        all_devices = [d for s in alloc["stages"] for d in s["devices"]]
        
        stage_ranks, current_rank = [], 0
        for stage in alloc["stages"]:
            ranks = list(range(current_rank, current_rank + len(stage["devices"])))
            stage_ranks.append(ranks)
            current_rank += len(stage["devices"])
            
        global_rank = 0
        for stage_idx, stage in enumerate(alloc["stages"]):
            for dev_idx, device_ip in enumerate(stage["devices"]):
                node_config = {
                    "my_rank": global_rank,
                    "my_stage_idx": stage_idx,
                    "stage_ranks": stage_ranks,
                    "layers_per_stage": [s["num_layers"] for s in alloc["stages"]],
                    "tp_ratios": stage["tp_ratios"],
                    "model_path": args.model,
                    "master_addr": alloc["global_root_ip"].split(':')[0],
                    "master_port": 29501,
                    "world_size": len(all_devices),
                    "num_tasks": args.num_tasks,
                    "disable_tp": args.disable_tp
                }
                if global_rank == 0: my_config = node_config
                else: send_config_to_node(device_ip, args.port, node_config)
                global_rank += 1

    if not my_config: return

    os.environ['MASTER_ADDR'] = my_config['master_addr']
    os.environ['MASTER_PORT'] = str(my_config['master_port'])
    dist.init_process_group(backend="gloo", rank=my_config['my_rank'], world_size=my_config['world_size'])
    
    tp_group = None
    for ranks in my_config['stage_ranks']:
        if len(ranks) > 1:
            g = dist.new_group(ranks)
            if my_config['my_rank'] in ranks: tp_group = g

    model_config = ModelConfig() # Default config
    
    # Auto-detect config from model file if available
    detected_cfg = read_m_file_config(my_config.get('model_path'))
    if detected_cfg:
        for k, v in detected_cfg.items():
            if hasattr(model_config, k):
                setattr(model_config, k, v)
                
    model_config.disable_tp = my_config.get('disable_tp', False)
    if model_config.disable_tp and my_config['my_rank'] == 0:
        logger.warning("TP All-Reduce is DISABLED for performance testing! Results may be incorrect.")
        
    model = OptimizedDistributedQwen3Model(model_config, my_config, tp_group=tp_group)
    model.load_weights(my_config['model_path'])
    
    # --- Task Initialization ---
    tasks = []
    tokenizer = None
    
    # Define prompts for multi-task pipeline testing
    all_prompts = [
        "The capital of France is",
        "Artificial Intelligence will", 
        "Python is popular because",
        "The future of space exploration",
        "Deep learning transforms"
    ]
    # Limit to 5 tasks as requested
    # Use config from rank 0 if available, otherwise default to args or 1
    num_tasks = my_config.get('num_tasks', args.num_tasks)
    num_tasks = min(num_tasks, 5)
    if num_tasks < 1: num_tasks = 1
    prompts = all_prompts[:num_tasks]
    
    if my_config['my_rank'] == 0:
        if args.tokenizer:
            # Check if .t file (Custom)
            if args.tokenizer.endswith('.t'):
                 logger.info(f"Custom tokenizer file detected: {args.tokenizer}")
                 tokenizer = None
                 # 1. Try loading as a standard format (renamed)
                 try:
                     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
                     logger.info("Successfully loaded .t file using AutoTokenizer.")
                 except Exception as e:
                     logger.warning(f"Could not load .t file with AutoTokenizer: {e}")
                 
                 # 2. Fallback to Tiktoken if failed
                 if tokenizer is None:
                     logger.info("Using Tiktoken fallback for .t file.")
                     try:
                        import tiktoken
                        class TiktokenWrapper:
                            def __init__(self):
                                self.enc = tiktoken.get_encoding("cl100k_base")
                            def encode(self, text, return_tensors=None):
                                ids = self.enc.encode(text)
                                if return_tensors == "pt":
                                    return torch.tensor([ids], dtype=torch.long)
                                return ids
                            def decode(self, ids):
                                if isinstance(ids, torch.Tensor): ids = ids.tolist()
                                valid_ids = []
                                for i in ids:
                                    try:
                                        self.enc.decode([i])
                                        valid_ids.append(i)
                                    except Exception: pass
                                return self.enc.decode(valid_ids)
                        tokenizer = TiktokenWrapper()
                     except ImportError:
                        logger.error("Tiktoken not found. Cannot load custom tokenizer.")
                        tokenizer = None
            else:
                # Try loading HuggingFace tokenizer
                try:
                    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
                except Exception as e:
                    logger.error(f"Failed to load HF tokenizer: {e}")
                    
                    # Fallback: Try raw tiktoken if available
                    try:
                        import tiktoken
                        logger.info("Falling back to raw tiktoken (cl100k_base)...")
                        class TiktokenWrapper:
                            def __init__(self):
                                self.enc = tiktoken.get_encoding("cl100k_base")
                            def encode(self, text, return_tensors=None):
                                ids = self.enc.encode(text)
                                if return_tensors == "pt":
                                    return torch.tensor([ids], dtype=torch.long)
                                return ids
                            def decode(self, ids):
                                if isinstance(ids, torch.Tensor): ids = ids.tolist()
                                valid_ids = []
                                for i in ids:
                                    try:
                                        self.enc.decode([i])
                                        valid_ids.append(i)
                                    except Exception: pass
                                return self.enc.decode(valid_ids)
                        tokenizer = TiktokenWrapper()
                    except ImportError:
                        logger.error("Tiktoken not found either. Using Dummy.")
                        tokenizer = None
        
        if tokenizer:
            print(f"\n[Rank 0] Initializing {len(prompts)} tasks for Pipeline Parallelism...")
            for i, p in enumerate(prompts):
                print(f"  Task {i}: {p}")
                tasks.append(TaskState(i, p, tokenizer=tokenizer))
        else:
            # Dummy task for testing without tokenizer
            print(f"\n[Rank 0] Initializing dummy tasks")
            for i in range(len(prompts)):
                tasks.append(TaskState(i, "Dummy", input_ids=torch.tensor([[1, 2, 3]])))
    else:
        # Other ranks initialize dummy tasks to match the count
        # They don't need prompts, just the task structure
        for i in range(len(prompts)):
            tasks.append(TaskState(i, "", input_ids=None))

    # Identify the rank that will broadcast the sampled token (Root of the last stage)
    last_stage_idx = len(my_config['stage_ranks']) - 1
    token_src_rank = my_config['stage_ranks'][last_stage_idx][0]

    model.profiler.reset()
    inference_start_time = time.perf_counter()
    MAX_NEW_TOKENS = 20
    
    # Loop until all tasks are complete
    # Pipelining Strategy:
    # We want to keep all stages busy.
    # Stage 0 should push tasks as fast as possible.
    # We iterate tasks in a round-robin fashion.
    
    step_count = 0
    
    while True:
        # Optimization: Filter active tasks BEFORE starting forward passes
        # This list must be consistent across all ranks.
        # We assume 'tasks' list is static and we only check is_complete flag.
        active_tasks = [t for t in tasks if not t.is_complete]
        if not active_tasks: break
        
        step_start_time = time.perf_counter()
        
        # Determine schedule: Which tasks to run in this micro-batch?
        # Simple schedule: Run ALL active tasks in round-robin order.
        # This naturally fills the pipeline.
        # Task 0 -> Stage 0 -> Stage 1 -> ...
        # Task 1 -> Stage 0 -> Stage 1 -> ...
        # When Stage 0 is done with Task 0, it starts Task 1 immediately.
        
        # 1. Forward Pass
        # We need to collect outputs from the LAST stage to broadcast.
        # But intermediate stages just forward.
        
        step_logits = {} 
        
        # NOTE: For proper pipeline sync, ALL ranks must iterate through active_tasks
        # in the exact same order.
        for task in active_tasks:
            # Profiling start
            if my_config['my_rank'] == 0 and task.token_count == 0 and task.start_time == 0.0:
                task.start_time = time.perf_counter()

            # Execute Forward
            if my_config['my_rank'] == 0:
                logits = model(task.curr_input_ids, start_pos=task.start_pos, task_id=task.task_id)
            else:
                logits = model(None, start_pos=task.start_pos, task_id=task.task_id)

            if my_config['my_rank'] == token_src_rank:
                step_logits[task.task_id] = logits

        # 2. Sample and Broadcast (Synchronization Point)
        # To minimize bubbles, we should ideally broadcast per task AS SOON AS it's ready.
        # But batching broadcast reduces overhead.
        # Let's keep batch broadcast for now, as network latency dominates.
        
        num_active = len(active_tasks)
        if num_active > 0:
            # Payload shape: [num_active, 4]
            # [token, step_len, task_id, is_complete]
            batch_payload = torch.zeros((num_active, 4), dtype=torch.long)
            
            if my_config['my_rank'] == token_src_rank:
                for i, task in enumerate(active_tasks):
                    # Check if we have logits. 
                    # If this rank is token_src_rank (last stage), it MUST have logits 
                    # UNLESS the pipeline is deeper than 1 and tasks are still in flight?
                    # In our synchronous-per-task model implementation:
                    # model() call blocks until it receives from previous stage, computes, and sends to next.
                    # So if token_src_rank finishes model(), it HAS the result.
                    # The only case it might fail is if an exception occurred earlier.
                    
                    if task.task_id not in step_logits:
                         # This should logically not happen in a blocking pipeline
                         logger.error(f"Missing logits for task {task.task_id}")
                         continue
                         
                    logits = step_logits[task.task_id]
                    next_token_val = torch.argmax(logits[:, -1, :], dim=-1).item()
                    step_len = logits.shape[1]
                    is_task_complete = 1 if task.token_count + 1 >= MAX_NEW_TOKENS else 0
                    
                    batch_payload[i, 0] = next_token_val
                    batch_payload[i, 1] = step_len
                    batch_payload[i, 2] = task.task_id
                    batch_payload[i, 3] = is_task_complete

            if HAS_DISTRIBUTED:
                # Ensure all ranks participate in broadcast!
                # If active_tasks differs between ranks, this will hang.
                # Since we update is_complete based on this broadcast, 
                # all ranks should stay in sync.
                try:
                    dist.broadcast(batch_payload, src=token_src_rank)
                except RuntimeError as e:
                    logger.error(f"Broadcast failed: {e}")
                    break # Abort if communication breaks
            
            # Unpack and Update
            for i, task in enumerate(active_tasks):
                token_val = batch_payload[i, 0].item()
                step_len_val = batch_payload[i, 1].item()
                task_id_val = batch_payload[i, 2].item()
                is_complete_val = batch_payload[i, 3].item()
                
                # Sanity check
                if task_id_val != task.task_id:
                      # This can happen if ranks have different active_tasks lists!
                      # Critical error.
                      logger.warning(f"Task ID mismatch! Rank {my_config['my_rank']} expects {task.task_id}, got {task_id_val}")
                
                task.start_pos += step_len_val
                task.token_count += 1
                if is_complete_val == 1: task.is_complete = True
                
                if my_config['my_rank'] == 0:
                    task.generated_ids.append(token_val)
                    task.curr_input_ids = torch.tensor([[token_val]], dtype=torch.long)
                    if task.is_complete:
                        task.end_time = time.perf_counter()
                        print(f"\n[Rank 0] Task {task.task_id} Completed! Duration: {task.end_time - task.start_time:.2f}s")
                        if tokenizer:
                            output_text = tokenizer.decode(task.generated_ids)
                            print(f"[Rank 0] Generated Text:\n{output_text}\n")
            
            step_end_time = time.perf_counter()
            if my_config['my_rank'] == 0:
                print(f"[Rank 0] Step {step_count}: Processed {num_active} tasks in {step_end_time - step_start_time:.4f}s (Avg {(step_end_time - step_start_time)/num_active:.4f}s/token)")
            step_count += 1


    # Print Profiler Stats for ALL ranks
    print(f"\n[Rank {my_config['my_rank']}] Performance Stats:")
    model.profiler.print_stats()

    if HAS_DISTRIBUTED:
        dist.destroy_process_group()

def test_local_2_devices(args):
    # 2 Devices: 1@14 * 1@14
    model_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B-Q4_0/dllama_model_qwen3_0.6b_q40.m"
    # Point to the directory containing tokenizer.json (Standard Qwen Tokenizer)
    tok_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B-Q4_0/dllama_tokenizer_qwen3_0.6b.t"
    ips = "127.0.0.1:30500,127.0.0.1:30501"
    config = "1@14*1@14"
    
    extra = []
    if args.disable_tp: extra.append("--disable_tp")
    if args.num_tasks > 1: extra.extend(["--num_tasks", str(args.num_tasks)])
    
    # Process 1 (Listener)
    p1 = mp.Process(target=main_wrapper, args=("listen", "127.0.0.1", 30501, None, None, None, None, []))
    # Process 0 (Assigner)
    p0 = mp.Process(target=main_wrapper, args=("assign", "127.0.0.1", 30500, config, ips, model_path, tok_path, extra))
    
    p1.start(); time.sleep(1); p0.start()
    p0.join(); p1.join()

def main_wrapper(mode, ip, port, config, ips, model_path, tok_path, extra_args=[]):
    # Wrapper to set sys.argv and call main
    sys.argv = ["prog", "--mode", mode, "--my_ip", ip, "--port", str(port)]
    if mode == "assign":
        sys.argv.extend(["--config", config, "--ips", ips, "--model", model_path, "--tokenizer", tok_path])
        if extra_args:
            sys.argv.extend(extra_args)
    main()

def test_local_4_devices():
    # 4 Devices: 1:1@14 * 1:1@14 (2 stages, TP=2 each)
    model_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B-Q4_0/dllama_model_qwen3_0.6b_q40.m"
    tok_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B"
    ips = "127.0.0.1:29600,127.0.0.1:29601,127.0.0.1:29602,127.0.0.1:29603"
    config = "1:1@14*1:1@14"
    
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

if __name__ == "__main__":
    main()
