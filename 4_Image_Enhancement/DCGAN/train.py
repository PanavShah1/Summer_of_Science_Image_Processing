import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import Discriminator, Generator, initialize_weights

def train():

    # Hyperparameters etc.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # From the paper --
    LEARNING_RATE = 2e-4  
    BATCH_SIZE = 128
    IMAGE_SIZE = 64
    CHANNELS_IMG = 3 # 1 for MNIST
    Z_DIM = 100
    NUM_EPOCHS = 1
    FEATURES_DISC = 64
    FEATURES_GEN = 64

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
            )
        ]
    )

    # dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
    dataset = datasets.ImageFolder(root="dataset/celeb_dataset", transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
    gen.load_state_dict(torch.load("gen_celeb.pth"))
    disc.load_state_dict(torch.load("disc_celeb.pth"))
    # initialize_weights(gen)
    # initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

    writer_real = SummaryWriter(f"logs_celeb/real") # logs/real
    writer_fake = SummaryWriter(f"logs_celeb/fake") # logs/fake
    step = 0

    gen.train()
    disc.train()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, _) in tqdm(enumerate(dataloader)):
            real = real.to(device)
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)

            ### Train Discriminator max log(D(x)) + log(1 - D(G(x)))
            disc_real = disc(real).reshape(-1) 
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

            ### Train Generator min log(1 - D(G(x))) <--> max log(D(G(x)))
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Print losses occasionally and print to tensorboard
            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                    Loss D: {loss_disc}, loss G: {loss_gen}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1

    torch.save(gen.state_dict(), "gen_celeb.pth") # gen.pth
    torch.save(disc.state_dict(), "disc_celeb.pth") # disc.pth

if __name__ == "__main__":
    train()
