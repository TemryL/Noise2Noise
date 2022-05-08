import torch
from model import Model
import matplotlib.pyplot as plt
import numpy as np

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
    device = model.device
    
    # Load training set:
    noisy_imgs_1 , noisy_imgs_2 = torch.load('../Data/train_data.pkl')
    noisy_imgs_1 = noisy_imgs_1.float()
    noisy_imgs_2 = noisy_imgs_2.float()
    noisy_imgs_1 = noisy_imgs_1.to(device)
    noisy_imgs_2 = noisy_imgs_2.to(device)
    
    # Load validation set:
    noisy_imgs, clean_imgs = torch.load('../Data/val_data.pkl')
    noisy_imgs = noisy_imgs.float()
    clean_imgs = clean_imgs.float()
    
    # Normalize data:
    pixel_normalization(noisy_imgs_1)
    pixel_normalization(noisy_imgs_2)
    pixel_normalization(noisy_imgs)
    pixel_normalization(clean_imgs)
    
    # Training:
    train = False
    if train:
        model.train(noisy_imgs_1.narrow(0, 0, 49920), noisy_imgs_2.narrow(0, 0, 49920), 100)
        torch.save(model.state_dict(), 'bestmodel.pth')
    
    # Validation:
    model = model.to('cpu')
    model.load_pretrained_model()
    
    psn_ratio = 0
    count = 1
    denoised_imgs = []
    for noisy, clean in zip(noisy_imgs, clean_imgs):
        denoised_imgs.append(model.predict(noisy[None, :, :, :]))
        psn_ratio += psnr(denoised_imgs[-1], clean[None, :, :, :])
    
    psn_ratio = psn_ratio/len(denoised_imgs)
    print("Current model achieves {} dB PSNR on the validation dataset".format(psn_ratio))
    
    # Plot result
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1, 3):
        sample_idx = torch.randint(len(noisy_imgs), size=(1,)).item()
        
        noisy = np.array(noisy_imgs[sample_idx])
        figure.add_subplot(rows, cols, i)
        plt.title('Noisy')
        plt.axis("off")
        plt.imshow(np.transpose(noisy.squeeze(), (1, 2, 0)), cmap="gray")
        
        denoised = np.array(denoised_imgs[sample_idx].detach()) 
        figure.add_subplot(rows, cols, i+1)
        plt.title('Denoised')
        plt.axis("off")
        plt.imshow(np.transpose(denoised.squeeze(), (1, 2, 0)), cmap="gray")
        
        clean = np.array(clean_imgs[sample_idx])
        figure.add_subplot(rows, cols, i+2)
        plt.title('Clean')
        plt.axis("off")
        plt.imshow(np.transpose(clean.squeeze(), (1, 2, 0)), cmap="gray")
    
    plt.show()

