
import struct
import os

file_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B-Q4_0/dllama_model_qwen3_0.6b_q40.m"

def inspect_layer_data():
    vocab_size = 151936
    dim = 1024
    
    # Calculate offset for first layer
    # Header: 136
    # Embeddings: vocab * dim * 4 (float32)
    offset = 136 + (vocab_size * dim * 4)
    
    print(f"Calculated Offset for Layer 0: {offset}")
    
    with open(file_path, 'rb') as f:
        f.seek(0, 2)
        total = f.tell()
        print(f"Total size: {total}")
        
        f.seek(offset)
        # Read some bytes
        data = f.read(64)
        print("Layer 0 Data (First 64 bytes):")
        print([b for b in data])
        
        # Read next few bytes to check for potential block scaling structure
        # In GGUF Q4_0: [scale(f16), byte, byte...]
        # In Q8_0: [scale(f16)? or just bytes?]
        
        # Let's interpret as floats or ints
        print("\nInterpreting as Int8:")
        print([int(b) if b < 128 else int(b) - 256 for b in data])

inspect_layer_data()
