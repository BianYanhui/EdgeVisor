
import torch

def test_int8_matmul():
    print("Testing Int8 Matmul...")
    a = torch.randint(-127, 127, (32, 32), dtype=torch.int8)
    b = torch.randint(-127, 127, (32, 32), dtype=torch.int8)
    
    try:
        c = torch.matmul(a, b)
        print(f"Result Type: {c.dtype}")
        print("Success!")
    except Exception as e:
        print(f"Failed: {e}")

    try:
        # Try float32 simulated
        af = a.to(torch.float32)
        bf = b.to(torch.float32)
        cf = torch.matmul(af, bf)
        print("Float32 works (obviously)")
    except:
        pass

test_int8_matmul()
