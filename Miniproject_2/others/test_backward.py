from torch.nn import functional as F
import torch
from model import ReLU, Sequential, Conv2d, Sigmoid, MSE, SGD, Model
from torch import nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(4, 3, kernel_size=3, stride=2, padding=0, dilation=1)
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        self.criterion = torch.nn.MSELoss()
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.sigmoid(self.conv2(x))
        return x
    
    def train(self, train_input, train_target, num_epochs):
        # Normalize data
        train_input = train_input.div(255.0)
        train_target = train_target.div(255.0)
        
        mini_batch_size = 2
        for e in range(num_epochs):
            epoch_loss = 0
            for b in range(0, train_input.size(0), mini_batch_size):
                output = self(train_input.narrow(0, b, mini_batch_size))
                loss = self.criterion(output, train_target.narrow(0, b, mini_batch_size))
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print("Epoch {}: Loss {}".format(e, epoch_loss))

if __name__ == "__main__":
    
    batch_size = 20
    
    input = torch.randn((batch_size, 3, 32, 32)).mul(255.0)
    target = torch.randn((batch_size, 3, 7, 7)).mul(255.0)
    
    net = Net()
    mymodel = Model()
    
    net.conv1.weight.data = mymodel.model.modules[0].weight.mul(1.0) 
    net.conv1.bias.data = mymodel.model.modules[0].bias.mul(1.0) 
    
    actual = mymodel.model.modules[0].weight
    expected = net.conv1.weight.data
    
    print(actual.data_ptr() == expected.data_ptr()) 
    torch.testing.assert_allclose(expected, actual)
    
    # x = net(input)
    # y = mymodel(input)
    
    # torch.testing.assert_allclose(x, y)
    
    ############# Test Training #############
    print("-------------- Train Net --------------")
    net.train(input, target, 50)
    
    print("\n-------------- Train Model --------------")
    mymodel.train(input, target, 50)
    
    print(actual.data_ptr() == expected.data_ptr()) 
    torch.testing.assert_allclose(expected, actual)
    #########################################
    
    