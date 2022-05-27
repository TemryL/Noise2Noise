import torch
from model import Conv2d

x = torch.randn(3,32,32,32, requires_grad=True)
y = torch.randn(3,3,32,32, requires_grad=True)

conv1 = torch.nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)  
my_conv1 = Conv2d(32, 3, kernel_size=3, stride=1, padding=1)  

conv1.weight.data.copy_(my_conv1.weight)
conv1.bias.data.copy_(my_conv1.bias)

# Forward:
torch.testing.assert_allclose(conv1(x), my_conv1(x))

# Backward:
output = conv1(x)
output.retain_grad()
output.backward(y)

my_conv1.backward(y)

# print(conv1.weight.grad)
# print(my_conv1.weight.grad)
torch.testing.assert_allclose(conv1.weight.grad, my_conv1.weight.grad)

