import torch
from model import Model
import matplotlib.pyplot as plt
import torchvision as tv
import numpy as np

def psnr(denoised, ground_truth):
    # Peak Signal to Noise Ratio : denoised and ground_truth have range [0 , 1]
    mse = torch.mean((denoised - ground_truth)**2)
    return -10*torch.log10(mse + 10**(-8)).item()

def imshow(img, normal):
    if normal:
        npimg = img.detach().numpy()   # convert from tensor
    else:
        npimg = img.detach().numpy().astype('uint8')   # convert from tensor
    plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    plt.show()

def normalize(input):
    mu, std = input.mean(), input.std()
    input.sub_(mu).div_(std)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
train = False
normal = True

# Load training set:
noisy_imgs_1 , noisy_imgs_2 = torch.load('../Data/train_data.pkl')
noisy_imgs_1.to(device)
noisy_imgs_2.to(device)

noisy_imgs_1 = noisy_imgs_1.float()
noisy_imgs_2 = noisy_imgs_2.float()

if normal:
    normalize(noisy_imgs_1)
    normalize(noisy_imgs_2)

# Load validation set:
noisy_imgs, clean_imgs = torch.load('../Data/val_data.pkl')

noisy_imgs.to(device)
clean_imgs.to(device)

noisy_imgs = noisy_imgs.float()
clean_imgs = clean_imgs.float()

if normal:
    normalize(noisy_imgs)
    normalize(clean_imgs)

# Training:
if train:
    model.train(noisy_imgs_1.narrow(0, 0, 1000), noisy_imgs_2.narrow(0, 0, 1000), 100)
    torch.save(model.state_dict(), 'bestmodel.pth')

# Validation:
model.load_pretrained_model()

# for i in range(5):  
#     imshow(tv.utils.make_grid(clean_imgs[i]), normal)
#     imshow(tv.utils.make_grid(model(noisy_imgs[None, i, :, :])), normal)

psn_ratio = 0
count = 1
for noisy, clean in zip(noisy_imgs, clean_imgs):
    denoised = model(noisy[None, :, :, :])
    psn_ratio += psnr(denoised, clean[None, :, :, :])
    count += 1
    if count<5:
        imshow(tv.utils.make_grid(clean), normal)
        imshow(tv.utils.make_grid(denoised), normal)
    print(psnr(denoised, clean[None, :, :, :]))
psn_ratio = psn_ratio/count
print("Current model achieves {} dB PSNR on the validation dataset".format(psn_ratio))



