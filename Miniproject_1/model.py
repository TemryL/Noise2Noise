import torch
from torch import nn, optim
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self):
        # instantiate model + optimizer + loss function + any other stuff you need
        super().__init__()
        
        # # Encoder
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1)
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        # self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        # self.conv5 = nn.Conv2d(32, 8, kernel_size=4, stride=1)
        
        # # Decoder
        # self.conv6 = nn.ConvTranspose2d(8, 32, kernel_size=4, stride=1)
        # self.conv7 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2)
        # self.conv8 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2)
        # self.conv9 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1)
        # self.conv10 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1)
        
        # def forward(self, x):
        #     x = F.relu(self.conv1(x))
        #     x = F.relu(self.conv2(x))
        #     x = F.relu(self.conv3(x))
        #     x = F.relu(self.conv4(x))
        #     x = F.relu(self.conv5(x))
        #     x = F.relu(self.conv6(x))
        #     x = F.relu(self.conv7(x))
        #     x = F.relu(self.conv8(x))
        #     x = F.relu(self.conv9(x))
        #     x = F.relu(self.conv10(x))
        #     return x
        
        #Encoder
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)
        
        self.optimizer = optim.Adam(self.parameters(), lr = 1e-3)
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = torch.sigmoid(self.t_conv2(x))
        return x
    
    def load_pretrained_model(self):
        # This loads the parameters saved in bestmodel .pth into the model
        m_state_dict = torch.load('bestmodel.pth')
        self.load_state_dict(m_state_dict)
    
    def train(self, train_input, train_target, num_epochs):
        # train_input : tensor of size (N, C, H, W) containing a noisy version of the images
        # train_target : tensor of size (N, C, H, W) containing another noisy version of the 
        # same images , which only differs from the input by their noise .
        batch_size = 10
        for e in range(num_epochs):
            epoch_loss = 0
            for b in range(0, train_input.size(0), batch_size):
                output = self(train_input.narrow(0, b, batch_size))
                loss = self.criterion(output, train_target.narrow(0, b, batch_size))
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print("Epoch {}: Loss {}".format(e, epoch_loss))

    def predict(self, test_input):
        # test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained
        # or the loaded network .
        # returns a tensor of the size (N1 , C, H, W)
        return self(test_input)