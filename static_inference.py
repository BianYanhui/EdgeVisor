import os
import sys
import json
import argparse
import socket
import time
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
    def __init__(self, cfg, tp_rank, tp_world_size, tp_group=None):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size
        self.tp_group = tp_group
        self.hidden_dim = cfg.hidden_dim
        self.emb_dim = cfg.emb_dim
        
        self.local_hidden_dim = self.hidden_dim // tp_world_size
        
        self.fc1 = nn.Linear(cfg.emb_dim, self.local_hidden_dim, dtype=cfg.dtype, bias=False)
        self.fc2 = nn.Linear(cfg.emb_dim, self.local_hidden_dim, dtype=cfg.dtype, bias=False)
        self.fc3 = nn.Linear(self.local_hidden_dim, cfg.emb_dim, dtype=cfg.dtype, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x_intermediate = nn.functional.silu(x_fc1) * x_fc2
        x_out = self.fc3(x_intermediate)
        
        if self.tp_world_size > 1:
            dist.all_reduce(x_out, op=dist.ReduceOp.SUM, group=self.tp_group)
            
        return x_out

class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg, tp_rank, tp_world_size, tp_group=None):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size
        self.tp_group = tp_group
        self.num_heads = cfg.n_heads
        self.num_kv_groups = cfg.n_kv_groups
        self.head_dim = cfg.head_dim
        
        self.local_num_heads = self.num_heads // tp_world_size
        self.local_kv_groups = self.num_kv_groups // tp_world_size
        
        self.d_out = self.local_num_heads * self.head_dim
        self.d_kv = self.local_kv_groups * self.head_dim
        
        self.W_query = nn.Linear(cfg.emb_dim, self.d_out, bias=True, dtype=cfg.dtype)
        self.W_key = nn.Linear(cfg.emb_dim, self.d_kv, bias=True, dtype=cfg.dtype)
        self.W_value = nn.Linear(cfg.emb_dim, self.d_kv, bias=True, dtype=cfg.dtype)
        self.out_proj = nn.Linear(self.d_out, cfg.emb_dim, bias=False, dtype=cfg.dtype)
        
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6) if cfg.qk_norm else None
        
        self.k_cache = None
        self.v_cache = None

    def forward(self, x, mask, cos, sin, start_pos=0):
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
        if start_pos == 0:
            # First step: initialize cache
            self.k_cache = keys
            self.v_cache = values
        else:
            # Subsequent steps: append to cache
            # Note: We need to handle the case where start_pos resets (new generation)
            # A simple way is to check if start_pos matches cache length
            if self.k_cache is not None and start_pos == self.k_cache.shape[2]:
                self.k_cache = torch.cat([self.k_cache, keys], dim=2)
                self.v_cache = torch.cat([self.v_cache, values], dim=2)
            else:
                # Reset or mismatch, re-initialize (should ideally not happen in correct flow)
                # But for safety, if start_pos < cache size, we truncate or reset.
                # Here we assume standard generation flow.
                if self.k_cache is not None and start_pos < self.k_cache.shape[2]:
                     self.k_cache = self.k_cache[:, :, :start_pos, :]
                     self.v_cache = self.v_cache[:, :, :start_pos, :]
                     self.k_cache = torch.cat([self.k_cache, keys], dim=2)
                     self.v_cache = torch.cat([self.v_cache, values], dim=2)
                else:
                    self.k_cache = keys
                    self.v_cache = values
        
        # Use cached keys/values for attention
        # keys/values: [b, n_kv_groups, seq_len, head_dim]
        keys = self.k_cache
        values = self.v_cache
        
        if self.local_kv_groups > 0:
            local_group_size = self.local_num_heads // self.local_kv_groups
            keys = keys.repeat_interleave(local_group_size, dim=1)
            values = values.repeat_interleave(local_group_size, dim=1)
        
        attn_scores = queries @ keys.transpose(2, 3)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        out = self.out_proj(context)
        
        if self.tp_world_size > 1:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
            
        return out

class TransformerBlock(nn.Module):
    def __init__(self, cfg, tp_rank, tp_world_size, tp_group=None):
        super().__init__()
        self.att = GroupedQueryAttention(cfg, tp_rank, tp_world_size, tp_group)
        self.ff = FeedForward(cfg, tp_rank, tp_world_size, tp_group)
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

class StaticDistributedQwen3Model(nn.Module):
    def __init__(self, config, my_config, tp_group=None):
        super().__init__()
        self.config = config
        self.tp_group = tp_group
        self.my_rank = my_config['my_rank']
        self.world_size = my_config['world_size']
        self.my_stage_idx = my_config['my_stage_idx']
        self.stage_ranks = my_config['stage_ranks']
        self.layers_per_stage = my_config['layers_per_stage']
        
        self.tp_group_ranks = self.stage_ranks[self.my_stage_idx]
        self.tp_world_size = len(self.tp_group_ranks)
        self.tp_rank = self.tp_group_ranks.index(self.my_rank)
        
        self.start_layer = sum(self.layers_per_stage[:self.my_stage_idx])
        self.num_layers = self.layers_per_stage[self.my_stage_idx]
        
        logger.info(f"[Rank {self.my_rank}] Stage {self.my_stage_idx}, TP {self.tp_rank}/{self.tp_world_size}, Layers {self.start_layer}-{self.start_layer+self.num_layers-1}")
        
        self.layers = nn.ModuleList([
            TransformerBlock(config, self.tp_rank, self.tp_world_size, self.tp_group) 
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
                chunk = w.shape[dim] // self.tp_world_size
                start = self.tp_rank * chunk
                if dim == 0: target.data.copy_(w[start:start+chunk])
                else: target.data.copy_(w[:, start:start+chunk])

            load_split(block.att.W_query.weight, f"{prefix}self_attn.q_proj.weight", 0)
            load_split(block.att.W_query.bias, f"{prefix}self_attn.q_proj.bias", 0)
            load_split(block.att.W_key.weight, f"{prefix}self_attn.k_proj.weight", 0)
            load_split(block.att.W_key.bias, f"{prefix}self_attn.k_proj.bias", 0)
            load_split(block.att.W_value.weight, f"{prefix}self_attn.v_proj.weight", 0)
            load_split(block.att.W_value.bias, f"{prefix}self_attn.v_proj.bias", 0)
            load_split(block.att.out_proj.weight, f"{prefix}self_attn.o_proj.weight", 1)
            
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

    def forward(self, x, start_pos=0):
        # Stage > 0: Receive from Prev Stage
        if self.my_stage_idx > 0:
            prev_root = self.stage_ranks[self.my_stage_idx - 1][0]
            if self.tp_rank == 0:
                shape = torch.zeros(3, dtype=torch.long)
                dist.recv(shape, src=prev_root)
                b, s, d = shape.tolist()
                x = torch.zeros((b, s, d), dtype=self.config.dtype)
                dist.recv(x, src=prev_root)
                sp = torch.zeros(1, dtype=torch.long)
                dist.recv(sp, src=prev_root)
                start_pos = sp.item()
            else:
                x = None
                start_pos = 0
            
            # Broadcast to TP group
            if self.tp_world_size > 1:
                if self.tp_rank == 0:
                    shape = torch.tensor(x.shape, dtype=torch.long)
                    sp = torch.tensor([start_pos], dtype=torch.long)
                else:
                    shape = torch.zeros(3, dtype=torch.long)
                    sp = torch.zeros(1, dtype=torch.long)
                
                dist.broadcast(shape, src=self.stage_ranks[self.my_stage_idx][0], group=self.tp_group)
                if self.tp_rank != 0:
                    b, s, d = shape.tolist()
                    x = torch.zeros((b, s, d), dtype=self.config.dtype)
                dist.broadcast(x, src=self.stage_ranks[self.my_stage_idx][0], group=self.tp_group)
                dist.broadcast(sp, src=self.stage_ranks[self.my_stage_idx][0], group=self.tp_group)
                start_pos = sp.item()

        # Stage 0: Embeddings
        if self.my_stage_idx == 0:
            if self.tp_world_size > 1:
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
            
            x = self.tok_emb(x)

        # Layers
        b, seq, _ = x.shape
        mask = None
        if seq > 1:
            mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1).to(x.device)
            
        for layer in self.layers:
            x = layer(x, mask, self.cos, self.sin, start_pos=start_pos)
            
        # Send to Next Stage or Return Logits
        if self.my_stage_idx < len(self.stage_ranks) - 1:
            if self.tp_rank == 0:
                next_root = self.stage_ranks[self.my_stage_idx + 1][0]
                dist.send(torch.tensor(x.shape, dtype=torch.long), dst=next_root)
                dist.send(x, dst=next_root)
                dist.send(torch.tensor([start_pos], dtype=torch.long), dst=next_root)
            return None
        else:
            x = self.final_norm(x)
            if self.tp_world_size > 1:
                # If TP > 1, hidden states might be split or need aggregation if coming from a split layer
                # But here, TransformerBlock output is already aggregated if it was split (e.g. FFN output AllReduce)
                # However, if we split the Norm or something else, we might need synchronization.
                # In this implementation, FFN and Attn outputs are AllReduced within the block, 
                # so x entering final_norm is fully aggregated and identical across TP ranks.
                pass
            
            # Only Rank 0 of the last stage computes logits to save computation
            if self.tp_rank == 0:
                return self.out_head(x)
            else:
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
    parser.add_argument("--test_2", action="store_true")
    parser.add_argument("--test_4", action="store_true")
    args = parser.parse_args()

    if args.test_2:
        test_local_2_devices()
        return
    if args.test_4:
        test_local_4_devices()
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
    
    if my_config['my_rank'] == 0:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        prompt = args.prompt if args.prompt else "Hello"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(torch.long)
        
        print(f"\n[Rank 0] Input: {prompt}")
        print(f"[Rank 0] Generating...")
        
        curr_input_ids = input_ids
        generated_ids = input_ids[0].tolist()
        start_pos = 0
    else:
        curr_input_ids = None
        start_pos = 0
        generated_ids = []

    # Identify the rank that will broadcast the sampled token (Root of the last stage)
    last_stage_idx = len(my_config['stage_ranks']) - 1
    token_src_rank = my_config['stage_ranks'][last_stage_idx][0]

    for i in range(20):
        # 1. Run Model Step
        if my_config['my_rank'] == 0:
            # Stage 0: Feed input (prompt or token)
            logits = model(curr_input_ids, start_pos=start_pos)
        else:
            # Other Stages: Receive input from prev stage
            # If Last Stage: returns logits
            # If Middle Stage: returns None
            logits = model(None, start_pos=start_pos)

        # 2. Sample and Broadcast (Token Generation)
        next_token = torch.zeros(1, 1, dtype=torch.long)
        
        # Only the designated rank (Last Stage, TP Rank 0) samples
        if my_config['my_rank'] == token_src_rank:
            if logits is None:
                # Should not happen for this rank
                logger.error("Logits is None on token source rank!")
                break
                
            # Sample from logits
            # logits: [batch, seq, vocab]
            next_token_val = torch.argmax(logits[:, -1, :], dim=-1) # [batch]
            next_token[0, 0] = next_token_val.item()
        
        # Broadcast the token to all ranks
        if HAS_DISTRIBUTED:
            dist.broadcast(next_token, src=token_src_rank)
        
        # 3. Update State for Next Step
        
        # Sync step_len (input length) from Rank 0 to all ranks to update start_pos correctly
        step_len = 0
        if my_config['my_rank'] == 0:
            step_len = curr_input_ids.shape[1]
            
        if HAS_DISTRIBUTED:
            sl_tensor = torch.tensor([step_len], dtype=torch.long)
            dist.broadcast(sl_tensor, src=0)
            step_len = sl_tensor.item()
            
        start_pos += step_len

        if my_config['my_rank'] == 0:
            token_id = next_token.item()
            generated_ids.append(token_id)
            print(tokenizer.decode([token_id]), end='', flush=True)
            
            # Prepare next input
            curr_input_ids = next_token

    if my_config['my_rank'] == 0:
        print("\n")
            
    if HAS_DISTRIBUTED:
        dist.destroy_process_group()

def _run_test_process(mode, ip, port, config, ips, model_path, tok_path):
    sys.argv = ["static_inference.py", "--mode", mode, "--my_ip", ip, "--port", str(port)]
    if mode == "assign":
        sys.argv.extend(["--config", config, "--ips", ips, "--model", model_path, "--tokenizer", tok_path])
    main()

def test_local_2_devices():
    # 2 Devices: 1@12 * 1@12 (2 stages, no TP)
    model_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B/model.safetensors"
    tok_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B"
    ips = "127.0.0.1,127.0.0.1:29501"
    config = "1@12*1@12"
    
    p1 = mp.Process(target=_run_test_process, args=("listen", "127.0.0.1", 29501, None, None, None, None))
    p0 = mp.Process(target=_run_test_process, args=("assign", "127.0.0.1", 29500, config, ips, model_path, tok_path))
    
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

if __name__ == "__main__":
    main()
