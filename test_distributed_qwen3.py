import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import json
from distributed_qwen3 import DistributedQwen3Model, DistributedConfig, LocalKVCache, RMSNorm, compute_rope_params, apply_rope

# --- Reference Model (Single Process) ---

class RefFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)

class RefGroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg["n_heads"]
        self.num_kv_groups = cfg["n_kv_groups"]
        self.head_dim = cfg["head_dim"]
        self.group_size = self.num_heads // self.num_kv_groups
        self.d_out = self.num_heads * self.head_dim
        
        # Enable bias for Q, K, V, disable for out_proj
        self.W_query = nn.Linear(cfg["emb_dim"], self.d_out, bias=True, dtype=cfg["dtype"])
        self.W_key = nn.Linear(cfg["emb_dim"], self.num_kv_groups * self.head_dim, bias=True, dtype=cfg["dtype"])
        self.W_value = nn.Linear(cfg["emb_dim"], self.num_kv_groups * self.head_dim, bias=True, dtype=cfg["dtype"])
        self.out_proj = nn.Linear(self.d_out, cfg["emb_dim"], bias=False, dtype=cfg["dtype"])
        
        if cfg["qk_norm"]:
            self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        
        if self.q_norm: queries = self.q_norm(queries)
        if self.k_norm: keys = self.k_norm(keys)
        
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys = apply_rope(keys, cos, sin, offset=start_pos)
        
        # Expand K, V
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)
        
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)

class RefTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = RefGroupedQueryAttention(cfg)
        self.ff = RefFeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin, start_pos) + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x) + shortcut
        return x

class ReferenceQwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.ModuleList([RefTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        
        head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(head_dim, theta_base=cfg["rope_base"], context_length=cfg["context_length"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        x = self.tok_emb(x)
        b, seq, _ = x.shape
        mask = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1).bool()[None, None, :, :]
        
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
            
        x = self.final_norm(x)
        return self.out_head(x)

    def load_weights_from_hf(self, model_path):
        from safetensors import safe_open
        
        # Load index if exists
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        weight_files = []
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)
            weight_files = sorted(list(set(index["weight_map"].values())))
        else:
            if os.path.exists(os.path.join(model_path, "model.safetensors")):
                weight_files = ["model.safetensors"]
            else:
                files = os.listdir(model_path)
                weight_files = [f for f in files if f.endswith(".safetensors")]
                
        print(f"Reference Model Loading weights from {weight_files}")
        
        for w_file in weight_files:
            file_path = os.path.join(model_path, w_file)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    self._load_single_weight(key, f.get_tensor(key))
                    
    def _load_single_weight(self, key, tensor):
        def assign(param, data, name):
            if param.shape != data.shape:
                print(f"Shape Mismatch for {name}: Param {param.shape} vs Data {data.shape}")
                # Try transpose?
                if param.shape == data.T.shape:
                    print("Transposing data...")
                    data = data.T
                else:
                    raise ValueError(f"Shape mismatch for {name}")
            
            with torch.no_grad():
                param.copy_(data)

        if "model.layers" in key:
            parts = key.split(".")
            layer_idx = int(parts[2])
            block = self.trf_blocks[layer_idx]
            module_name = parts[3]
            param_type = parts[-1] # "weight" or "bias"
            
            if module_name == "self_attn":
                proj = parts[4]
                target_module = None
                if proj == "q_proj": target_module = block.att.W_query
                elif proj == "k_proj": target_module = block.att.W_key
                elif proj == "v_proj": target_module = block.att.W_value
                elif proj == "o_proj": target_module = block.att.out_proj
                
                if target_module is not None:
                    if param_type == "bias" and target_module.bias is not None:
                        assign(target_module.bias, tensor, key)
                    elif param_type == "weight":
                        assign(target_module.weight, tensor, key)

            elif module_name == "mlp":
                proj = parts[4]
                target_module = None
                if proj == "gate_proj": target_module = block.ff.fc1
                elif proj == "up_proj": target_module = block.ff.fc2
                elif proj == "down_proj": target_module = block.ff.fc3
                
                if target_module is not None:
                    if param_type == "bias" and target_module.bias is not None:
                        assign(target_module.bias, tensor, key)
                    elif param_type == "weight":
                        assign(target_module.weight, tensor, key)

            elif module_name == "input_layernorm":
                if param_type == "weight":
                    assign(block.norm1.scale, tensor, key)
            elif module_name == "post_attention_layernorm":
                if param_type == "weight":
                    assign(block.norm2.scale, tensor, key)
                    
        elif "model.embed_tokens" in key:
            if "weight" in key:
                assign(self.tok_emb.weight, tensor, key)
                # Handle tied weights for output head
                assign(self.out_head.weight, tensor, key)
        elif "model.norm" in key:
            if "weight" in key:
                assign(self.final_norm.scale, tensor, key)
        elif "lm_head" in key:
            if "weight" in key:
                assign(self.out_head.weight, tensor, key)

# --- Test Runner ---

def run_distributed_process(rank, world_size, cfg, model_path, expected_logits_path, input_ids):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(42)

    # Config
    stage_ranks = [[0, 1], [2, 3, 4]]
    tp_ranks_per_stage = [2, 3]
    dist_config = DistributedConfig(rank, world_size, stage_ranks, tp_ranks_per_stage)
    dist_config.setup_groups()

    # Determine Loaded Layers (Overlapping for Dynamic Test)
    # Stage 0: 0-12 (12 layers: 0..11, plus overlap? Let's load 0-12 to be safe for 12 layers)
    # Default split is 12, 12. 
    # Stage 0 needs 0-11.
    # Stage 1 needs 12-23.
    # We want to test [10, 14].
    # Stage 0 needs 0-9. (Subset of 0-11).
    # Stage 1 needs 10-23. (Needs 10, 11 which are usually in Stage 0).
    # So Stage 1 MUST load 10-23.
    
    if dist_config.stage_id == 0:
        managed_layers_range = [0, 12] # Loads 0-11
    else:
        managed_layers_range = [10, 24] # Loads 10-23 (Includes 10, 11)

    # Initialize Distributed Model
    model = DistributedQwen3Model(cfg, dist_config, managed_layers_range=managed_layers_range)
    
    # Load Weights
    model.load_weights_from_hf(model_path)
    
    local_kv = LocalKVCache(len(model.trf_blocks)) # Size based on loaded layers
    
    model.eval()
    
    # Log role
    print(f"Rank {rank}: Role = Stage {dist_config.stage_id}, TP Rank {dist_config.tp_rank}, Loaded Layers {managed_layers_range}")
    
    with torch.no_grad():
        # Generate 10 tokens
        generated_ids = []
        curr_input = input_ids
        current_pos = 0
        
        # Dynamic Schedule
        # First 3 tokens: [12, 12] (Explicit)
        # Next 3 tokens: None (Implicit fallback to [12, 12])
        # Next 3 tokens: [10, 14] (Explicit switch)
        # Last 1 token: None (Implicit fallback to [10, 14])
        
        layer_counts = [12, 12] # Initial default
        
        for step in range(10):
            runtime_counts = None
            
            if step == 0:
                runtime_counts = [12, 12]
                layer_counts = runtime_counts
            elif step == 6:
                runtime_counts = [10, 14]
                layer_counts = runtime_counts
                
            if rank == 0:
                print(f"Step {step}: Providing Runtime Counts: {runtime_counts} (Expected Active: {layer_counts})")
            
            logits = model(curr_input, cache=local_kv, runtime_layer_counts=runtime_counts, start_pos=current_pos)
            
            # Only last stage gets logits
            next_token_tensor = torch.zeros(1, 1, dtype=torch.long)
            
            if dist_config.is_last_stage and dist_config.tp_rank == 0:
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids.append(next_token.item())
                next_token_tensor[0, 0] = next_token
                
            # Broadcast next token (Sync)
            # Use Rank 2 (First rank of Last Stage) as source for simplicity in this topology
            sender = 2 
            dist.broadcast(next_token_tensor, src=sender)
            
            current_pos += curr_input.shape[1]
            curr_input = next_token_tensor
            
    # Verify Final Logic (Check consistency on the first generated token logits with SHIFTED config)
    dist.barrier()
    
    # Re-run single step for consistency check with [10, 14] split
    print(f"Rank {rank}: Verifying consistency with [10, 14] split...")
    local_kv = LocalKVCache(len(model.trf_blocks))
    with torch.no_grad():
        logits = model(input_ids, cache=local_kv, runtime_layer_counts=[10, 14])
        
    if dist_config.is_last_stage and dist_config.tp_rank == 0:
        expected = torch.load(expected_logits_path)
        diff = (logits - expected).abs().max()
        print(f"Rank {rank}: Max Difference (Shifted [10, 14]) = {diff.item()}")
        if diff < 1e-3:
            print("Rank {}: SUCCESS! Output matches reference.".format(rank))
        else:
            print("Rank {}: FAILURE! Output mismatch.".format(rank))
            
        # Decode generated ids
        if len(generated_ids) > 0:
            print(f"Rank {rank}: Generated IDs: {generated_ids}")


    dist.barrier()
    dist.destroy_process_group()

def main():
    import sys
    import subprocess

    def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    try:
        import numpy
    except ImportError:
        print("Installing numpy...")
        install("numpy")
    
    try:
        import safetensors
    except ImportError:
        print("Installing safetensors...")
        install("safetensors")

    try:
        import transformers
    except ImportError:
        print("Installing transformers...")
        install("transformers")

    from transformers import AutoTokenizer

    # User provided local path
    model_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B"
    
    # Inferred Config (since config.json is missing or unreliable)
    # Based on infer_config.py output:
    # vocab_size: 151936, emb_dim: 896, n_layers: 24, n_heads: 7, n_kv_groups: 1, head_dim: 128, hidden_dim: 4864
    cfg = {
        "vocab_size": 151936,
        "emb_dim": 896,
        "n_layers": 24,
        "n_heads": 7,
        "n_kv_groups": 1, # MQA
        "head_dim": 128,
        "hidden_dim": 4864,
        "dtype": torch.float32, 
        "rope_base": 1000000.0, # Default for Qwen2.5/3 usually
        "context_length": 32768,
        "qk_norm": False # Checked via script, no q_norm keys
    }
    
    print(f"Using Config: {cfg}")
    
    # Load Tokenizer
    print("Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        text = "Hello world"
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        print(f"Input: '{text}' -> IDs: {input_ids}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        # Fallback to dummy if tokenizer fails, but we should try hard
        input_ids = torch.tensor([[9707, 1879, 374, 264, 1279]], dtype=torch.long)
        print(f"Using dummy input ids: {input_ids}")

    print("Generating Reference Output (Single Process)...")
    ref_model = ReferenceQwen3Model(cfg)
    ref_model.load_weights_from_hf(model_path)
    
    with torch.no_grad():
        expected_logits = ref_model(input_ids)
        
    torch.save(expected_logits, "expected_logits.pt")
    
    print("Starting Distributed Test...")
    world_size = 5
    # Pass input_ids to distributed process
    mp.spawn(run_distributed_process, args=(world_size, cfg, model_path, "expected_logits.pt", input_ids), nprocs=world_size, join=True)
    
    if os.path.exists("expected_logits.pt"): os.remove("expected_logits.pt")

if __name__ == "__main__":
    main()
