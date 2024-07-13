import os
from gan import Generator
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
z_dim = 64 # 128, 256
img_dim = 28*28*1 # 784


gen = Generator(z_dim, img_dim).to(device)
gen.load_state_dict(torch.load("gen.pth"))

gen.eval()


with torch.inference_mode():
    for i in range(9):
        noise = torch.randn((1, z_dim)).to(device)
        gen_img = gen(noise)
        gen_img = gen_img.view(28, 28).detach().cpu().numpy()
        plt.subplot(3, 3, i+1)
        plt.imshow(gen_img, cmap="gray")

plt.show()
