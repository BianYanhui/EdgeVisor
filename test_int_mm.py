
import torch

def test_int_mm():
    try:
        a = torch.randint(-127, 127, (32, 32), dtype=torch.int8)
        b = torch.randint(-127, 127, (32, 32), dtype=torch.int8)
        
        # Check for _int_mm
        if hasattr(torch, '_int_mm'):
             c = torch._int_mm(a, b)
             print(f"_int_mm result type: {c.dtype}")
             print(f"Value check: {c[0,0]} vs {float(a[0,:] @ b[:,0].float())}")
        else:
             print("_int_mm not found")
             
    except Exception as e:
        print(f"Error: {e}")

test_int_mm()
