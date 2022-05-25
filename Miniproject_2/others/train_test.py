import torch
from model import Model
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)  
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.tconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2)
        self.tconv2 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2)
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        self.criterion = torch.nn.MSELoss()
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.tconv1(x))
        x = torch.sigmoid(self.tconv2(x))
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
        # print("-------------- Train Net --------------")
        # net.train(noisy_imgs_1.narrow(0, 0, 992), noisy_imgs_2.narrow(0, 0, 992), 50)
        
        print("\n-------------- Train Model --------------")
        model.train(noisy_imgs_1.narrow(0, 0, 9984), noisy_imgs_2.narrow(0, 0, 9984), 5)
        torch.save(model.param(), 'bestmodel.pth')
    
    # Validation:
    # model = model.to('cpu')
    model.load_pretrained_model()
    
    psn_ratio = 0
    count = 1
    denoised_imgs = []
    for noisy, clean in zip(noisy_imgs, clean_imgs):
        denoised_imgs.append(net(noisy[None, :, :, :]).div(255.0))
        psn_ratio += psnr(denoised_imgs[-1], clean[None, :, :, :].div(255.0))
    
    psn_ratio = psn_ratio/len(denoised_imgs)
    print("Net achieves {} dB PSNR on the validation dataset".format(psn_ratio))
    
    
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
