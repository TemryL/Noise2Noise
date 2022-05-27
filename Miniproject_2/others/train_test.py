import torch
from model import Model
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import torch
from torch import nn, optim
from pathlib import Path

class Net(nn.Module):
    def __init__(self):
        # instantiate model + optimizer + loss function + any other stuff you need
        super().__init__()
        
        #Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=2)  
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2, stride=2)
        
        #Decoder
        # self.t_conv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2)
        # self.t_conv2 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2)
        
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        
        self.optimizer = optim.Adam(self.parameters(), lr = 1e-3)
        self.criterion = nn.MSELoss()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(self.up(x)))
        x = torch.sigmoid(self.conv4(self.up(x)))
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
        
        mini_batch_size = 50
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
    
    def predict(self, test_input):
        # test_input : tensor of size (N1 , C, H, W) with values in range 0-255 that has to be denoised by the trained
        # or the loaded network .
        # returns a tensor of the size (N1 , C, H, W) with values in range 0-255.
        
        test_input = test_input.div(255.0)
        test_output = self(test_input).mul(255.0)
        return test_output


def psnr(denoised, ground_truth):
    # Peak Signal to Noise Ratio : denoised and ground_truth have range [0 , 1]
    mse = torch.mean((denoised - ground_truth)**2)
    return -10*torch.log10(mse + 10**(-8)).item()

def pixel_standardization(input):
    mu, std = input.mean(), input.std()
    input.sub_(mu).div_(std)

def pixel_normalization(input):
    input.div_(255.0)

if __name__ == '__main__':
    model = Model()
    net = Net()
    
    # Load training set:
    noisy_imgs_1 , noisy_imgs_2 = torch.load('../../Data/train_data.pkl')
    noisy_imgs_1 = noisy_imgs_1.float()
    noisy_imgs_2 = noisy_imgs_2.float()
    
    # Load validation set:
    noisy_imgs, clean_imgs = torch.load('../../Data/val_data.pkl')
    noisy_imgs = noisy_imgs.float()
    clean_imgs = clean_imgs.float()
    
    # Training:
    train = True
    if train:
        print("-------------- Train Net --------------")
        net.train(noisy_imgs_1.narrow(0, 0, 1000), noisy_imgs_2.narrow(0, 0, 1000), 1)
        
        print("\n-------------- Train Model --------------")
        model.train(noisy_imgs_1.narrow(0, 0, 1000), noisy_imgs_2.narrow(0, 0, 1000), 5)
        torch.save(model.param(), 'bestmodel.pth')
    
    # Validation:
    # model = model.to('cpu')
    #model.load_pretrained_model()
    
    psn_ratio = 0
    count = 1
    denoised_imgs = []
    for noisy, clean in zip(noisy_imgs, clean_imgs):
        denoised_imgs.append(net.predict(noisy[None, :, :, :]).div(255.0))
        psn_ratio += psnr(denoised_imgs[-1], clean[None, :, :, :].div(255.0))
    
    psn_ratio = psn_ratio/len(denoised_imgs)
    print("Net model achieves {} dB PSNR on the validation dataset".format(psn_ratio))
    
    psn_ratio = 0
    count = 1
    denoised_imgs = []
    for noisy, clean in zip(noisy_imgs, clean_imgs):
        denoised_imgs.append(model.predict(noisy[None, :, :, :]).div(255.0))
        psn_ratio += psnr(denoised_imgs[-1], clean[None, :, :, :].div(255.0))
    
    psn_ratio = psn_ratio/len(denoised_imgs)
    print("Current model achieves {} dB PSNR on the validation dataset".format(psn_ratio))
    
    # Plot result
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1, 3):
        sample_idx = torch.randint(len(noisy_imgs), size=(1,)).item()
        
        noisy = np.array(noisy_imgs[sample_idx].div(255.0))
        figure.add_subplot(rows, cols, i)
        plt.title('Noisy')
        plt.axis("off")
        plt.imshow(np.transpose(noisy.squeeze(), (1, 2, 0)), cmap="gray")
        
        denoised = np.array(denoised_imgs[sample_idx].detach()) 
        figure.add_subplot(rows, cols, i+1)
        plt.title('Denoised')
        plt.axis("off")
        plt.imshow(np.transpose(denoised.squeeze(), (1, 2, 0)), cmap="gray")
        
        clean = np.array(clean_imgs[sample_idx].div(255.0))
        figure.add_subplot(rows, cols, i+2)
        plt.title('Clean')
        plt.axis("off")
        plt.imshow(np.transpose(clean.squeeze(), (1, 2, 0)), cmap="gray")
    
    plt.show()