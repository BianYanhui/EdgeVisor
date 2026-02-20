
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from distributed_qwen3 import DistributedQwen3Model, DistributedConfig

# Configuration
MODEL_PATH = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B/model.safetensors"
TOKENIZER_PATH = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B"
WORLD_SIZE = 2
MAX_NEW_TOKENS = 20

def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(42)

    # 1. Setup Distributed Config (Manual 2-Stage Pipeline)
    # Stage 0: Rank 0 (Layers 0-11)
    # Stage 1: Rank 1 (Layers 12-23)
    stage_ranks = [[0], [1]]
    layers_per_stage = [12, 12] # Total 24 layers for Qwen-3-0.6B
    
    # Model Config (Hardcoded for Qwen-3-0.6B / Qwen2-0.5B)
    # Based on standard Qwen2-0.5B parameters since config.json is missing
    hf_config = {
        "vocab_size": 151936,
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "intermediate_size": 4864,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 32768
    }
        
    dist_config = DistributedConfig(
        vocab_size=hf_config["vocab_size"],
        emb_dim=hf_config["hidden_size"],
        n_layers=hf_config["num_hidden_layers"],
        n_heads=hf_config["num_attention_heads"],
        n_kv_groups=hf_config["num_key_value_heads"],
        head_dim=hf_config["hidden_size"] // hf_config["num_attention_heads"],
        hidden_dim=hf_config["intermediate_size"],
        rope_base=hf_config["rope_theta"],
        context_length=hf_config["max_position_embeddings"],
        dtype=torch.float32, # Use float32 for CPU/Gloo to avoid bfloat16 issues
        qk_norm=False # Verified via safetensors keys
    )

    # Initialize Model
    print(f"[Rank {rank}] Initializing model...")
    model = DistributedQwen3Model(dist_config, stage_ranks, layers_per_stage, rank, world_size)
    
    # Load Weights
    model.load_weights(MODEL_PATH)
    
    # 2. Inference Loop
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        prompt = "you are a good person with"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(torch.long)
        
        print(f"\n[Rank 0] Input: {prompt}")
        print(f"[Rank 0] Generating...")
        
        # --- Inference Loop (Stateless) ---
        # Since DistributedQwen3Model doesn't have KV Cache, we must pass the full sequence every time.
        curr_input_ids = input_ids.clone()
        generated_ids = input_ids[0].tolist()
        
        for i in range(MAX_NEW_TOKENS):
            # Forward pass with full sequence (sends to Rank 1)
            # start_pos=0 because we are processing the full sequence from scratch
            model(curr_input_ids, start_pos=0)
            
            # Receive next token from Rank 1
            next_token = torch.zeros(1, dtype=torch.long)
            dist.recv(next_token, src=1)
            
            # Update sequence
            curr_input_ids = torch.cat([curr_input_ids, next_token.unsqueeze(0)], dim=1)
            generated_ids.append(next_token.item())
            
            print(tokenizer.decode([next_token.item()]), end="", flush=True)
            
        print("\n\n[Rank 0] Full Generation:")
        print(tokenizer.decode(generated_ids))
        
    else: # Rank 1
        # Loop to serve requests
        # Total steps = 1 (prefill) + (MAX_NEW_TOKENS - 1) = MAX_NEW_TOKENS
        
        for i in range(MAX_NEW_TOKENS):
            # Wait for input from Rank 0 (via model.forward internal recv)
            # Returns logits because this is the last stage
            logits = model(None) 
            
            # Greedy Sampling
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            
            # Send back to Rank 0
            dist.send(next_token.cpu(), dst=0)

    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(run_worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
