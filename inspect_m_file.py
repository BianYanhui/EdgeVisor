
import struct
import os

file_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B-Q4_0/dllama_model_qwen3_0.6b_q40.m"

def inspect_file():
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'rb') as f:
        # Read Header (136 bytes as per code)
        header_bytes = f.read(136)
        config = {}
        print("Header Values:")
        for i in range(8, 136, 8):
            try:
                key, val = struct.unpack('<II', header_bytes[i:i+8])
                config[key] = val
                print(f"  Key {key}: {val}")
            except:
                pass
        
        # Read some data bytes
        print("\nFirst 64 bytes of data (after header):")
        data = f.read(64)
        print([b for b in data])

        # Check total file size
        f.seek(0, 2)
        total_size = f.tell()
        print(f"\nTotal File Size: {total_size} bytes")

inspect_file()
