import struct

file_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/Yanhui/杂乱/Models/Qwen-3-0.6B-Q4_0/dllama_model_qwen3_0.6b_q40.m"

def inspect_data():
    with open(file_path, 'rb') as f:
        f.seek(136) # Skip header
        data = f.read(16)
        
    print("Hex:", data.hex())
    
    # Try FP32
    floats = struct.unpack('<4f', data)
    print("FP32:", floats)
    
    # Try FP16
    # Python struct doesn't support 'e' (half float) until 3.6? It does.
    try:
        halves = struct.unpack('<8e', data)
        print("FP16:", halves)
    except:
        print("FP16 unpack failed")
        
    # Try Int8
    ints = struct.unpack('<16b', data)
    print("Int8:", ints)

if __name__ == "__main__":
    inspect_data()
