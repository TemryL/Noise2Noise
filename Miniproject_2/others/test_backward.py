from torch.nn import functional as F
import torch
from model import ReLU, Sequential, Conv2d, Sigmoid, MSE, SGD, Model
from torch import nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=4, stride=2)  
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=2)
        self.tconv1 = nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2)
        self.tconv2 = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2)
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1)
        self.criterion = torch.nn.MSELoss()
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
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
    
    
    input = torch.randn((1000, 3, 32, 32)).mul(255.0)
    target = torch.randn((1000, 3, 32, 32)).mul(255.0)
    
    net = Net()
    mymodel = Model()
    
    ############# Test Training #############
    # print("-------------- Train Net --------------")
    # net.train(input, target, 50)
    
    print("\n-------------- Train Model --------------")
    mymodel.train(input, target, 10)
    
    
    