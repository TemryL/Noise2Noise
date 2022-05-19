from tkinter import SE
import torch
from model import ReLU, Sequential, Conv2d
from torch import nn

if __name__ == "__main__":
    
    in_channels = 3
    out_channels = 4
    kernel_size = (3, 3)
    stride = (2,6)
    padd = (2, 3)
    dil = (2,4)
    
    conv1 = Conv2d(in_channels, out_channels, kernel_size, stride, padding=padd, dilation=dil)
    conv2 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padd, dilation=dil)
    
    conv1.weight = conv2.weight.data
    conv1.bias = conv2.bias.data
    
    batch_size = 3
    x = torch.randn((batch_size, in_channels, 32, 32))
    
    # Output of PyTorch convolution
    expected = conv2(x)
    
    # Output of convolution as a matrix product
    actual = conv1(x)
    
    torch.testing.assert_allclose(actual, expected)