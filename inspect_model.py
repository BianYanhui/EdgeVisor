import struct
import os

file_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B-Q4_0/dllama_model_qwen3_0.6b_q40.m"

def inspect_header():
    with open(file_path, 'rb') as f:
        header = f.read(256)
        
    print(f"File size: {os.path.getsize(file_path)}")
    
    # Try parsing standard llama2.c header (usually starts with magic or config)
    # The previous code assumed config starts at offset 8?
    # Let's look at the first few integers
    ints = struct.unpack('<64I', header)
    print("Header integers:", ints[:16])
    
    # Config dict mapping from optimized_inference.py
    # key 2: dim, 9: vocab, 4: layers
    # The file seems to use a key-value pair format in the header?
    # "key, val = struct.unpack('<II', header_bytes[i:i+8])"
    
    print("\nParsing Key-Value Header:")
    config = {}
    for i in range(8, 136, 8):
        key = ints[i//4]
        val = ints[i//4 + 1]
        config[key] = val
        print(f"Key {key}: {val}")

if __name__ == "__main__":
    inspect_header()
