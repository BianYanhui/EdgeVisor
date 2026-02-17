from safetensors import safe_open
import torch

model_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B/model.safetensors"

def check_bias():
    with safe_open(model_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        
        # Check lm_head
        has_lm_head = "lm_head.weight" in keys
        print(f"Has lm_head: {has_lm_head}")
        
        # Check specific layers for bias
        layers_to_check = [
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.self_attn.k_proj.bias",
            "model.layers.0.self_attn.v_proj.bias",
            "model.layers.0.self_attn.o_proj.bias",
            "model.layers.0.mlp.gate_proj.bias",
            "model.layers.0.mlp.up_proj.bias",
            "model.layers.0.mlp.down_proj.bias",
            "model.layers.0.input_layernorm.bias",
            "model.layers.0.post_attention_layernorm.bias",
            "model.norm.bias"
        ]
        
        for layer in layers_to_check:
            if layer in keys:
                tensor = f.get_tensor(layer)
                print(f"{layer}: Shape {tensor.shape}")
            else:
                print(f"{layer}: Not found")

if __name__ == "__main__":
    check_bias()
