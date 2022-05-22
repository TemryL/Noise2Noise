from torch.nn import functional as F
import torch
from model import ReLU, Sequential, Conv2d, Sigmoid, MSE, SGD
from torch import nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=0, dilation=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

if __name__ == "__main__":
    
    batch_size = 2
    
    input = torch.randn((batch_size, 3, 32, 32))
    target = torch.randn((batch_size, 3, 28, 28))
    
    net = Net()
    print(net(input).shape)
    conv1 = Conv2d(3, 4, kernel_size=3, stride=1, padding=0, dilation=1)
    conv2 = Conv2d(4, 3, kernel_size=3, stride=1, padding=0, dilation=1)
    
    conv1.weight = net.conv1.weight.data.mul(1.0)
    conv1.weight.grad = conv1.weight.mul(0.0)
    conv1.bias = net.conv1.bias.data.mul(1.0)
    conv1.bias.grad = conv1.bias.mul(0.0)
    conv2.weight = net.conv2.weight.data.mul(1.0)
    conv2.weight.grad = conv2.weight.mul(0.0)
    conv2.bias = net.conv2.bias.data.mul(1.0)
    conv2.bias.grad = conv2.bias.mul(0.0)
    
    seq = Sequential(conv1, ReLU(), conv2, ReLU())
    ################################################
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=1)
    
    # zero the parameter gradients
    optimizer.zero_grad()
    
    # forward + backward + optimize
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    expected = net.conv2.weight.data
    
    ###############################################
    
    torch.set_grad_enabled(False)
    criterion = MSE()
    optimizer = SGD(seq.param(), lr=1)
    
    # zero the parameter gradients
    optimizer.zero_grad()
    
    # forward + backward + optimize
    output = seq(input)
    loss = criterion(output, target)
    seq.backward(criterion.backward())
    optimizer.step()
    
    actual = seq.modules[2].weight
    ###############################################
    
    # in2 = seq(input)
    
    print(expected.data_ptr() == actual.data_ptr()) # prints True
    torch.testing.assert_allclose(expected, actual)