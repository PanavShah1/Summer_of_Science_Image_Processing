import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from torch.utils.tensorboard import SummaryWriter

# Things to try:
# 1. What happens if you use larger network?
# 2. Better normalization with BatchNorm
# 3. Different learning rates for the generator and discriminator
# 4. Change architecture to CNN

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1), # Fake or Real
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim): # z_dim: noise dimension, img_dim: image dimension
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )
    
    def forward(self, x):
        return self.gen(x)

def train_gan():

    # Hyperparameters etc.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 3e-4
    z_dim = 64 # 128, 256
    img_dim = 28*28*1 # 784
    batch_size = 32
    num_epochs = 50

    disc = Discriminator(img_dim).to(device)
    gen = Generator(z_dim, img_dim).to(device)
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)
    transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))] # Mean, Std
    )
    dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt_disc = optim.Adam(disc.parameters(), lr=lr)
    opt_gen = optim.Adam(gen.parameters(), lr=lr)
    criterion = nn.BCELoss()
    print(os.getcwd())
    if not os.path.exists("runs/GAN_MNIST"):
        os.makedirs("runs/GAN_MNIST/real")
        os.makedirs("runs/GAN_MNIST/fake")
        print("Created log directories.")
    writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
    writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
    step = 0

    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]

            ### Train Discriminator: max log(D(real)) + log(1 - D(G(z)))
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            ### Train Generator: min log(1 - D(G(z))) <--> max log(D(G(z)))
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] \ " 
                    f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                    data = real.reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(data, normalize=True)
                    writer_fake.add_image(
                        "Mnist Fake Images", img_grid_fake, global_step=step
                    )

                    writer_real.add_image(  
                        "Mnist Real Images", img_grid_real, global_step=step
                    )
                
                    step += 1

    writer_fake.close()
    writer_real.close()

    torch.save(gen.state_dict(), "gen.pth")
    torch.save(disc.state_dict(), "disc.pth")

if __name__ == "__main__":
    train_gan()
