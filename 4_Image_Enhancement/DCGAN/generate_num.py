import os
from model import Generator
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
Z_DIM = 100
CHANNELS_IMG = 1 
FEATURES_GEN = 64


gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
gen.load_state_dict(torch.load("gen.pth"))

gen.eval()


with torch.inference_mode():
    for i in range(9):
        noise = torch.randn((1, Z_DIM, 1, 1)).to(device)
        gen_img = gen(noise)
        gen_img = gen_img.squeeze().detach().cpu().numpy()
        plt.subplot(3, 3, i+1)
        plt.imshow(gen_img, cmap="gray")

plt.show()
