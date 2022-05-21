from torch.nn import functional as F
import torch
from model import ReLU, Sequential, Conv2d, Sigmoid, MSE, SGD
from torch import nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=(3,3), padding=(2,3), dilation=(2,4))
        self.conv2 = nn.Conv2d(4, 3, kernel_size=3, stride=(2,2), padding=(2,3), dilation=(2,4))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.sigmoid(self.conv2(x))
        return x

if __name__ == "__main__":
    
    in_channels = 3
    out_channels = 4
    kernel_size = (3,3)
    stride = (3,3)
    padd = (2, 3)
    dil = (2,4)
    batch_size = 1
    
    input = torch.randn((batch_size, in_channels, 32, 32))
    target = torch.randn((batch_size, 3, 6, 4))
    
    net = Net()
    
    conv1 = Conv2d(3, 4, kernel_size=3, stride=(3,3), padding=(2,3), dilation=(2,4))
    conv2 = Conv2d(4, 3, kernel_size=3, stride=(2,2), padding=(2,3), dilation=(2,4))
    
    conv1.weight = net.conv1.weight.data.mul(1.0)
    conv1.weight.grad = conv1.weight.mul(0.0)
    conv1.bias = net.conv1.bias.data.mul(1.0)
    conv1.bias.grad = conv1.bias.mul(0.0)
    conv2.weight = net.conv2.weight.data.mul(1.0)
    conv2.weight.grad = conv2.weight.mul(0.0)
    conv2.bias = net.conv2.bias.data.mul(1.0)
    conv2.bias.grad = conv2.bias.mul(0.0)
    
    seq = Sequential(conv1, ReLU(), conv2, Sigmoid())
    ################################################
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    
    # zero the parameter gradients
    optimizer.zero_grad()
    
    # forward + backward + optimize
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    expected = net.conv1.bias.data
    
    ###############################################
    
    torch.set_grad_enabled(False)
    criterion = MSE()
    optimizer = SGD(seq.param(), lr=0.1)
    
    # zero the parameter gradients
    optimizer.zero_grad()
    
    # forward + backward + optimize
    output = seq(input)
    loss = criterion(output, target)
    seq.backward(criterion.backward())
    optimizer.step()
    
    actual = seq.modules[0].bias
    ###############################################
    
    # in2 = seq(input)
    
    print(expected.data_ptr() == actual.data_ptr()) # prints True
    torch.testing.assert_allclose(expected, actual)