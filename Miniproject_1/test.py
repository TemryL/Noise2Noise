import torch
from model import Model
import matplotlib.pyplot as plt
import torchvision as tv
import numpy as np

def psnr(denoised, ground_truth):
    # Peak Signal to Noise Ratio : denoised and ground_truth have range [0 , 1]
    mse = torch.mean((denoised - ground_truth)**2)
    return -10*torch.log10(mse + 10**(-8)).item()

def imshow(img):
    npimg = img.detach().mul(255).numpy().astype('uint8')   # convert from tensor
    plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    plt.show()

def pixel_standardization(input):
    mu, std = input.mean(), input.std()
    input.sub_(mu).div_(std)

def pixel_normalization(input):
    input.div_(255)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)

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
noisy_imgs = noisy_imgs.to(device)
clean_imgs = clean_imgs.to(device)

# Normalize data:
pixel_normalization(noisy_imgs_1)
pixel_normalization(noisy_imgs_2)
pixel_normalization(noisy_imgs)
pixel_normalization(clean_imgs)

# Training:
train = True
if train:
    model.train(noisy_imgs_1.narrow(0, 0, 1920), noisy_imgs_2.narrow(0, 0, 1920), 25)
    torch.save(model.state_dict(), 'bestmodel.pth')

# Validation:
model.load_pretrained_model()

psn_ratio = 0
count = 1
for noisy, clean in zip(noisy_imgs, clean_imgs):
    denoised = model(noisy[None, :, :, :])
    psn_ratio += psnr(denoised, clean[None, :, :, :])
    count += 1
    # if count>5 and count<10:
    #     imshow(tv.utils.make_grid(clean))
    #     imshow(tv.utils.make_grid(noisy))
    #     imshow(tv.utils.make_grid(denoised))
psn_ratio = psn_ratio/count
print("Current model achieves {} dB PSNR on the validation dataset".format(psn_ratio))



