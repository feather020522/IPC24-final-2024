import torch

print(torch.__version__)
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device="cpu")
print(my_tensor)
print(torch.cuda.is_available())