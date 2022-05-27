import torch
from torch import nn, optim
from torch.nn import functional as F
from pathlib import Path

class Model(nn.Module):
    def __init__(self):
        # instantiate model + optimizer + loss function + any other stuff you need
        super().__init__()
        
        #Encoder
        self.conv1 = nn.Conv2d(3, 100, kernel_size=3)  
        self.conv2 = nn.Conv2d(100, 100, kernel_size=3)
        
        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(100, 100, kernel_size=3)
        self.t_conv2 = nn.ConvTranspose2d(2*100, 3, kernel_size=3)
        
        self.optimizer = optim.Adam(self.parameters(), lr = 1e-3)
        self.criterion = nn.MSELoss()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  
        
    def forward(self, x):
        res = F.relu(self.conv1(x))
        x = F.relu(self.conv2(res))
        x = F.relu(self.t_conv1(x))
        x = torch.cat((res, x), dim=1) 
        x = torch.sigmoid(self.t_conv2(x))
        return x
    
    def load_pretrained_model(self):
        # This loads the parameters saved in bestmodel .pth into the model
        model_path = Path(__file__).parent/"bestmodel.pth"
        m_state_dict = torch.load(model_path, map_location=torch.device(self.device))
        self.load_state_dict(m_state_dict)
    
    def train(self, train_input, train_target, num_epochs):
            # train_input : tensor of size (N, C, H, W) containing a noisy version of the images, with values in range 0-255.
            # train_target : tensor of size (N, C, H, W) containing another noisy version of the 
            # same images , which only differs from the input by their noise, with values in range 0-255.
            
            # Normalize data
            train_input = train_input.div(255.0)
            train_target = train_target.div(255.0)
            
            #mini_batch_size = 8
            mini_batch_size = 1000
            for e in range(num_epochs):
                epoch_loss = 0
                for b in range(0, train_input.size(0), mini_batch_size):
                    print(b)
                    output = self(train_input.narrow(0, b, mini_batch_size))
                    loss = self.criterion(output, train_target.narrow(0, b, mini_batch_size))
                    epoch_loss += loss.item()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                print("Epoch {}: Loss {}".format(e, epoch_loss))
    
    def predict(self, test_input):
        # test_input : tensor of size (N1 , C, H, W) with values in range 0-255 that has to be denoised by the trained
        # or the loaded network .
        # returns a tensor of the size (N1 , C, H, W) with values in range 0-255.
        
        test_input = test_input.div(255.0)
        test_output = self(test_input).mul(255.0)
        return test_output