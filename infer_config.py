
import os
from safetensors import safe_open
import json

model_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B/model.safetensors"

try:
    with safe_open(model_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        
        # 1. n_layers
        layer_indices = set()
        for k in keys:
            if "model.layers." in k:
                try:
                    idx = int(k.split(".")[2])
                    layer_indices.add(idx)
                except:
                    pass
        n_layers = max(layer_indices) + 1 if layer_indices else 0
        
        # 2. emb_dim & vocab_size
        if "model.embed_tokens.weight" in keys:
            tensor = f.get_tensor("model.embed_tokens.weight")
            vocab_size, emb_dim = tensor.shape
        else:
            vocab_size, emb_dim = 0, 0
            
        # 3. FFN hidden_dim
        # model.layers.0.mlp.gate_proj.weight shape (hidden, emb)
        hidden_dim = 0
        if f"model.layers.0.mlp.gate_proj.weight" in keys:
             t = f.get_tensor("model.layers.0.mlp.gate_proj.weight")
             hidden_dim = t.shape[0]
             
        # 4. Heads
        # q_proj (n_heads * head_dim, emb)
        # k_proj (n_kv * head_dim, emb)
        n_heads = 0
        n_kv_groups = 0
        head_dim = 0
        
        if f"model.layers.0.self_attn.q_proj.weight" in keys:
            q_w = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
            k_w = f.get_tensor("model.layers.0.self_attn.k_proj.weight")
            
            q_out = q_w.shape[0]
            k_out = k_w.shape[0]
            
            # Usually head_dim is 64 or 128.
            # let's assume head_dim = emb_dim // n_heads? No.
            # Try to guess head_dim.
            # Common head_dim: 64, 128.
            # If q_out % 128 == 0: head_dim = 128?
            
            if q_out % 128 == 0 and k_out % 128 == 0:
                head_dim = 128
            elif q_out % 64 == 0 and k_out % 64 == 0:
                head_dim = 64
            else:
                head_dim = q_out // (emb_dim // 128) # fallback guess?
                
            n_heads = q_out // head_dim
            n_kv_groups = k_out // head_dim
            
        print(json.dumps({
            "vocab_size": vocab_size,
            "emb_dim": emb_dim,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_kv_groups": n_kv_groups,
            "head_dim": head_dim,
            "hidden_dim": hidden_dim
        }, indent=2))
        
except Exception as e:
    print(f"Error: {e}")
