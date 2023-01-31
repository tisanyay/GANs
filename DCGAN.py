import os 
import tensorflow as tf    
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchmetrics.image.fid import FrechetInceptionDistance

import matplotlib.pyplot as plt

class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(

        )

    def forward(self, x):
        x = x.view(-1, 3072)
        x = self.layers(x)
        return x

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 3072),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layers(x)
        return x.view(-1, 3, 32, 32)

class SimpleAgent():
    def __init__(self, latent_dim, batch_size, disc_lr=3e-4, gen_lr=3e-4):
        self.latent_dim = latent_dim
        self.criterion = nn.BCELoss()
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.latent_dim = latent_dim
        self.disc = DCGANDiscriminator().to(self.device)
        self.gen = DCGANGenerator(latent_dim).to(self.device)

        self.disc_optim = optim.Adam(self.disc.parameters(), disc_lr)
        self.gen_optim = optim.Adam(self.gen.parameters(), gen_lr)
        self.fid = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=True).to(self.device)

        self.disc_loss, self.gen_loss, self.real_acc, self.fake_acc = [], [], [], []
        self.test_disc_loss, self.test_gen_loss, self.test_real_acc, self.test_fake_acc = [], [], [], []
        self.epoch_disc_loss, self.epoch_gen_loss, self.epoch_real_acc, self.epoch_fake_acc, self.fid_score = [], [], [], [], []

    def train_disc(self, fake_imgs, real_imgs):
        disc_real = self.disc(real_imgs)
        disc_fake = self.disc(fake_imgs)

        real_acc = torch.sum(torch.round(disc_real) == torch.ones_like(disc_real)) / len(disc_real)
        fake_acc = torch.sum(torch.round(disc_fake) == torch.zeros_like(disc_fake)) / len(disc_fake)
    
        real_loss = self.criterion(disc_real, torch.ones_like(disc_real))
        fake_loss = self.criterion(disc_fake, torch.zeros_like(disc_fake))
        disc_loss = (real_loss + fake_loss) / 2
        self.disc.zero_grad()
        disc_loss.backward(retain_graph=True)
        self.disc_optim.step()

        self.epoch_disc_loss.append(disc_loss.item()) 
        self.epoch_real_acc.append(real_acc.item()) 
        self.epoch_fake_acc.append(fake_acc.item()) 

        return disc_loss.item(), real_acc.item(), fake_acc.item()

    def train_gen(self, fake_imgs):
        disc_fake = self.disc(fake_imgs)
        gen_loss = self.criterion(disc_fake, torch.ones_like(disc_fake))
        self.gen.zero_grad()
        gen_loss.backward()
        self.gen_optim.step()

        self.epoch_gen_loss.append(gen_loss.item())

        return gen_loss.item()

    def test_gan(self, real_imgs):
        self.disc.eval()
        self.gen.eval()
        with torch.no_grad():
            real_imgs = real_imgs.float().cuda()
            noise = torch.randn(self.latent_dim*self.batch_size).to(self.device)
            fake_imgs = self.gen(noise.reshape(self.batch_size, self.latent_dim))

            disc_real = self.disc(real_imgs)
            disc_fake = self.disc(fake_imgs)

            real_acc = torch.sum(torch.round(disc_real) == torch.ones_like(disc_real)) / len(disc_real)
            fake_acc = torch.sum(torch.round(disc_fake) == torch.zeros_like(disc_fake)) / len(disc_fake)

            real_loss = self.criterion(disc_real, torch.ones_like(disc_real))
            fake_loss = self.criterion(disc_fake, torch.zeros_like(disc_fake))
            disc_loss = (real_loss + fake_loss) / 2

            disc_fake = self.disc(fake_imgs)
            gen_loss = self.criterion(disc_fake, torch.ones_like(disc_fake))

            self.fid.update(fake_imgs, real=False)

            self.epoch_disc_loss.append(disc_loss.item())
            self.epoch_gen_loss.append(gen_loss.item())
            self.epoch_real_acc.append(real_acc.item())
            self.epoch_fake_acc.append(fake_acc.item())

            return disc_loss.item(), gen_loss.item(), real_acc.item(), fake_acc.item()

    def print_progress(self, num_batches, num_epochs, current_epoch, i):
        progress = int(((i+1)/num_batches)*20)
        print(f"{current_epoch+1}/{num_epochs} [{'='*progress}{' '*(20-progress)}] {i+1}/{num_batches}", end='\r', flush=True)

    def output_imgs(self, dir, disc_loss, gen_loss, fid, epoch):
        rows, columns = 8, 6
        noise = torch.randn(self.latent_dim*rows*columns).to(self.device)
        fake_imgs = self.gen(noise.reshape(rows*columns, self.latent_dim))
        fig, axs = plt.subplots(rows, columns, figsize=(20, 30))
        for i in range(rows*columns):
            ax = axs[i//columns, i%columns]
            img = (fake_imgs[i].cpu().permute(1,2,0).detach().numpy() + 1) / 2
            ax.imshow(img, cmap='Greys')
            ax.axis('off')
    
        plt.text(-192, -310, f'epoch: {epoch}   discriminator loss: {disc_loss:5.2f}   generator loss: {gen_loss:5.2f}   fid: {fid:5.2f}', fontsize=30, ha='left', va='top')
        plt.savefig(dir+f'/img{epoch}.jpg')
        plt.close()

    def output_test_graphs(self, dir):
        self.test_disc_loss.append(sum(self.epoch_disc_loss) / len(self.epoch_disc_loss))
        self.test_gen_loss.append(sum(self.epoch_gen_loss) / len(self.epoch_gen_loss))
        self.test_real_acc.append(sum(self.epoch_real_acc) / len(self.epoch_real_acc))
        self.test_fake_acc.append(sum(self.epoch_fake_acc) / len(self.epoch_fake_acc))

        self.epoch_gen_loss, self.epoch_disc_loss, self.epoch_real_acc, self.epoch_fake_acc = [], [], [], []

        fig = plt.figure(figsize=(15,9))
        plt.plot(self.test_disc_loss)
        plt.plot(self.test_gen_loss)
        plt.plot(self.fid_score)
        plt.legend(['discriminator loss', 'generator loss', 'fid'])
        plt.title('Discriminator loss vs Generator loss vs FID')
        plt.savefig(dir+'/test_loss_graph.jpg')
        plt.close()

        fig = plt.figure(figsize=(15,9))
        plt.plot(self.test_real_acc)
        plt.plot(self.test_fake_acc)
        plt.legend(['real accuracy', 'fake_accuracy'])
        plt.title('Real accuracy vs fake accuracy')
        plt.savefig(dir+'/test_acc_graph.jpg')
        plt.close()

        fid_score = self.fid.compute().item()
        self.fid.reset()
        self.fid_score.append(fid_score)

        return self.test_disc_loss[-1], self.test_gen_loss[-1], self.test_real_acc[-1], self.test_fake_acc[-1], fid_score

    def output_train_graphs(self, dir):
        self.disc_loss.append(sum(self.epoch_disc_loss) / len(self.epoch_disc_loss))
        self.gen_loss.append(sum(self.epoch_gen_loss) / len(self.epoch_gen_loss))
        self.real_acc.append(sum(self.epoch_real_acc) / len(self.epoch_real_acc))
        self.fake_acc.append(sum(self.epoch_fake_acc) / len(self.epoch_fake_acc))

        self.epoch_gen_loss, self.epoch_disc_loss, self.epoch_real_acc, self.epoch_fake_acc = [], [], [], []

        fig = plt.figure(figsize=(15,9))
        plt.plot(self.disc_loss)
        plt.plot(self.gen_loss)
        plt.legend(['discriminator loss', 'generator loss'])
        plt.title('Discriminator loss vs Generator loss')
        plt.savefig(dir+'/train_loss_graph.jpg')
        plt.close()

        fig = plt.figure(figsize=(15,9))
        plt.plot(self.real_acc)
        plt.plot(self.fake_acc)
        plt.legend(['real accuracy', 'fake_accuracy'])
        plt.title('Real accuracy vs fake accuracy')
        plt.savefig(dir+'/train_acc_graph.jpg')
        plt.close()