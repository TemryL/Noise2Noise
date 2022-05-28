import torch
from model import Model
import matplotlib.pyplot as plt
import numpy as np
import time

def psnr(denoised, ground_truth):
    # Peak Signal to Noise Ratio : denoised and ground_truth have range [0 , 1]
    mse = torch.mean((denoised - ground_truth)**2)
    return -10*torch.log10(mse + 10**(-8)).item()

if __name__ == '__main__':
    model = Model()
    
    ######################### Load training set: #########################
    noisy_imgs_1 , noisy_imgs_2 = torch.load('../Data/train_data.pkl')
    noisy_imgs_1 = noisy_imgs_1.float()
    noisy_imgs_2 = noisy_imgs_2.float()
    
    ######################### Load validation set: #########################
    noisy_imgs, clean_imgs = torch.load('../Data/val_data.pkl')
    noisy_imgs = noisy_imgs.float()
    clean_imgs = clean_imgs.float()
    
    ######################### Training: #########################
    train = True
    if train:
        print("Training model")
        start = time.time()
        model.train(noisy_imgs_1.narrow(0, 0, 50000), noisy_imgs_2.narrow(0, 0, 50000), 10)
        torch.save(model.param(), 'bestmodel.pth')
        end = time.time()
        print("Elapsed time: {}s".format(end-start))
    
    ######################### Validation: #########################
    model.load_pretrained_model()
    
    psn_ratio = 0
    count = 1
    denoised_imgs = []
    for noisy, clean in zip(noisy_imgs, clean_imgs):
        denoised_imgs.append(model.predict(noisy[None, :, :, :]).div(255.0))
        psn_ratio += psnr(denoised_imgs[-1], clean[None, :, :, :].div(255.0))
    
    psn_ratio = psn_ratio/len(denoised_imgs)
    print("Current model achieves {} dB PSNR on the validation dataset".format(psn_ratio))
    
    ######################### Plot results: #########################
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