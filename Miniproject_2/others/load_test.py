import torch

x = torch.randn(3,3)
print(x)

torch.save(x, "test.pth")
y = torch.load("test.pth")

print(y)