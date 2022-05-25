from torch.nn import functional as F
import torch
from model import ReLU, Sequential, Conv2d, Sigmoid, MSE, SGD, Model
from torch import nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=2)  
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=2)
        
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
        
        mini_batch_size = 32
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
    
    
    input = torch.randn((992, 3, 32, 32)).mul(255.0)
    target = torch.randn((992, 3, 7, 7)).mul(255.0)
    
    net = Net()
    mymodel = Model()
    
    net.conv1.weight.data.copy_(mymodel.model.modules[0].weight)
    net.conv1.bias.data.copy_(mymodel.model.modules[0].bias)
    net.conv2.weight.data.copy_(mymodel.model.modules[2].weight)
    net.conv2.bias.data.copy_(mymodel.model.modules[2].bias)
    
    expected_w1 = net.conv1.weight.data
    expected_w2 = net.conv2.weight.data
    expected_b1 = net.conv1.bias.data
    expected_b2 = net.conv2.bias.data
    
    actual_w1 = mymodel.model.modules[0].weight
    actual_w2 = mymodel.model.modules[2].weight
    actual_b1 = mymodel.model.modules[0].bias
    actual_b2 = mymodel.model.modules[2].bias
    
    print(expected_w1.data_ptr() == actual_w1.data_ptr())
    print(expected_w2.data_ptr() == actual_w2.data_ptr())
    print(expected_b1.data_ptr() == actual_b1.data_ptr())
    print(expected_b2.data_ptr() == actual_b2.data_ptr())
    
    torch.testing.assert_allclose(expected_w1, actual_w1)
    torch.testing.assert_allclose(expected_w2, actual_w2)
    torch.testing.assert_allclose(expected_b1, actual_b1)
    torch.testing.assert_allclose(expected_b2, actual_b2)
    
    #torch.testing.assert_allclose(net(input), mymodel(input))
    ############ Test Training #############
    print("-------------- Train Net --------------")
    net.train(input, target, 10)
    
    print("\n-------------- Train Model --------------")
    mymodel.train(input, target, 10)
    
    torch.testing.assert_allclose(expected_w1, actual_w1)
    torch.testing.assert_allclose(expected_w2, actual_w2)
    torch.testing.assert_allclose(expected_b1, actual_b1)
    torch.testing.assert_allclose(expected_b2, actual_b2)
    
    
    