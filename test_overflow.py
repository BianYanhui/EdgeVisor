
import torch

a = torch.tensor([[100]], dtype=torch.int8)
b = torch.tensor([[100]], dtype=torch.int8)
c = torch.matmul(a, b)
print(f"100 * 100 = {c.item()} (dtype: {c.dtype})")

a = torch.tensor([[100]], dtype=torch.int32)
b = torch.tensor([[100]], dtype=torch.int32)
c = torch.matmul(a, b)
print(f"100 * 100 (int32) = {c.item()}")
