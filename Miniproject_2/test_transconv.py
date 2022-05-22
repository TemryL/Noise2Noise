from torch.nn import functional as F
import torch
from model import ReLU, Sequential, Conv2d, Sigmoid, MSE, SGD, TransposeConv2d
from torch import nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.tconv1 = nn.ConvTranspose2d(3, 4, kernel_size=3, stride=1, padding=0, dilation=1)
    
    def forward(self, x):
        x = self.tconv1(x)
        return x
    
if __name__ == "__main__":
    
    batch_size = 2
    
    input = torch.randn((batch_size, 3, 32, 32))
    target = torch.randn((batch_size, 3, 28, 28))
    
    tconv1 = TransposeConv2d(3, 4, kernel_size=3, stride=1, padding=0, dilation=1)
    net = Net()
    # tconv1.weight = net.tconv1.weight.data.mul(1.0)
    # tconv1.bias = net.tconv1.bias.data.mul(1.0)
    
    # print(tconv1.weight.shape)
    # print(net.tconv1.weight.data.shape)
    
    # expected = net(input)
    # actual = tconv1(input)
    
    # print(expected.data_ptr() == actual.data_ptr()) # prints True
    # torch.testing.assert_allclose(expected, actual)
    
    output = tconv1(input)

    dx = tconv1.backward(output)
    print(tconv1.weight)