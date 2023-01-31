from utils import *
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

random_seed = 9292
torch.manual_seed(random_seed)
LATENT_DIM = 100
BATCH_SIZE = 64
NUM_EPOCHS = 100
AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS = int(os.cpu_count() / 2)
lr = 3e-4
latent_dims = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

class ACGANGenerator(nn.Module):
    def __init__(self, latent_dims):
        super(ACGANGenerator, self).__init__()
        self.latent_dims = latent_dims

        # first linear layer
        self.fc1 = nn.Linear(110, 384)
        # Transposed Convolution 2
        self.conv1= nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.conv2= nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.conv3= nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(48, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        input = input.view(-1, self.latent_dims)
        fc1 = self.fc1(input)
        fc1 = fc1.view(-1, 384, 1, 1)
        tconv2 = self.tconv2(fc1)
        tconv3 = self.tconv3(tconv2)
        tconv4 = self.tconv4(tconv3)
        tconv5 = self.tconv5(tconv4)
        output = tconv5
        return output


class ACGANDiscriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(ACGANDiscriminator, self).__init__()

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(4*4*512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(4*4*512, num_classes)
        # softmax and sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        flat6 = conv6.view(-1, 4*4*512)
        fc_dis = self.fc_dis(flat6)
        fc_aux = self.fc_aux(flat6)

        classes = self.softmax(fc_aux)
        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
        return realfake, classes

if __name__ == "__main__":
    from utils import *

    data_dir = './data'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    training = CIFAR10(data_dir, train=True, transform=transform, download=True)
    testing = CIFAR10(data_dir, train=False, transform=transform, download=True)

    train_dataloader = DataLoader(training, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(testing, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)



    disc = ACGANDiscriminator()
    gen = ACGANGenerator(LATENT_DIM)

    disc_optim = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    gen_optim = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))

    disc_criterion = nn.BCELoss()
    aux_criterion = nn.NLLLoss()



    def train_acgan(disc, gen, disc_optim, gen_optim, disc_criterion, aux_criterion, basedir, train_dataloader, test_dataloader):
        disc.train()
        gen.train()
        train_num_batches = len(train_dataloader)
        dict_result = {}
        dict_result['avg_disc_loss'], dict_result['avg_gen_loss'], dict_result['avg_real_acc'], dict_result['avg_fake_acc'] = [], [], [], []
        dict_result['test_disc_loss'], dict_result['test_gen_loss'], dict_result['test_real_acc'], dict_result['test_fake_acc'], dict_result['fid_score'] = [], [], [], [], []
        
        # test_conditional_gan(gen, disc, criterion, dict_result, 0, basedir, test_dataloader, fid)
        
        for epoch in range(NUM_EPOCHS):
            epoch_disc_loss, epoch_gen_loss, epoch_real_acc, epoch_fake_acc = [], [], [], []
            for i, (real_imgs, labels) in enumerate(train_dataloader):
                disc.zero_grad()
                random_labels = torch.randint(0, 10, (real_imgs.shape[0], )).to(device)
                # real disc training
                disc_real, classes_real = disc(real_imgs.to(device))

                _batch_size = labels.shape[0]
                
                labels = labels.resize(_batch_size).to(device)

                disc_real = disc_criterion(disc_real, torch.ones_like(disc_real))
                aux_real = aux_criterion(classes_real, labels)

                errD_real = disc_real + aux_real 
                errD_real.backward()

                accuracy = compute_acc(classes_real, labels)

                # fake disc training
                real_imgs = real_imgs.float().cuda()
                noise = torch.randn(LATENT_DIM*real_imgs.shape[0]).to(device)
                fake_labels = torch.one_hot(torch.randint(0, 10, (_batch_size, )), num_classes=10)
                fake_imgs = gen(noise.reshape(real_imgs.shape[0], LATENT_DIM))

                disc_fake, classes_fake = disc(fake_imgs)

                disc_fake = disc_criterion(disc_fake, torch.zeros_like(disc_fake))
                aux_fake = aux_criterion(classes_fake, fake_labels)
                errD_fake = disc_fake + aux_fake
                # real_acc = torch.sum(torch.round(disc_real) == torch.ones_like(disc_real)) / len(disc_real)
                # fake_acc = torch.sum(torch.round(disc_fake) == torch.zeros_like(disc_fake)) / len(disc_fake)
            
                # disc_loss = (real_loss + fake_loss) / 2
                errD_fake.backward(retain_graph=True)
                disc_optim.step()

                epoch_disc_loss.append(disc_loss.item()) 
                epoch_real_acc.append(real_acc.item()) 
                epoch_fake_acc.append(fake_acc.item()) 

                disc_fake, classes_fake = disc(fake_imgs)
                # gen_loss = criterion(disc_fake, torch.ones_like(disc_fake))
                gen_fake = disc_criterion(disc_fake, torch.ones_like(disc_fake))
                gen_aux = aux_criterion(classes_fake, fake_labels)
                gen_loss = gen_fake + gen_aux
                gen.zero_grad()
                gen_loss.backward()
                gen_optim.step()

                epoch_gen_loss.append(gen_loss.item())

                progress = int(((i+1)/train_num_batches)*20)
                print(f"{epoch+1}/{NUM_EPOCHS} [{'='*progress}{' '*(20-progress)}] {i+1}/{train_num_batches}", end='\r', flush=True)

                del disc_loss, gen_loss, real_acc, fake_acc, noise, fake_imgs, disc_real, disc_fake, real_loss, fake_loss, random_labels

            # dcgan_agent.output_train_graphs('.')
            
            avg_disc_loss = sum(epoch_disc_loss) / len(epoch_disc_loss)
            avg_gen_loss = sum(epoch_gen_loss) / len(epoch_gen_loss)
            avg_real_acc = sum(epoch_real_acc) / len(epoch_real_acc)
            avg_fake_acc = sum(epoch_fake_acc) / len(epoch_fake_acc)

            dict_result['avg_disc_loss'].append(avg_disc_loss)
            dict_result['avg_gen_loss'].append(avg_gen_loss)
            dict_result['avg_real_acc'].append(avg_real_acc)
            dict_result['avg_fake_acc'].append(avg_fake_acc)

            del epoch_disc_loss, epoch_gen_loss, epoch_real_acc, epoch_fake_acc

            plot_training_graphs(basedir, dict_result['avg_disc_loss'], dict_result['avg_gen_loss'], dict_result['avg_real_acc'], dict_result['avg_fake_acc'])

            print(f'\ndisc loss: {avg_disc_loss:5.2f}  gen loss: {avg_gen_loss:5.2f}  real acc: {avg_real_acc:5.2f}  fake_acc: {avg_fake_acc:5.2f}\n')
            del avg_disc_loss, avg_gen_loss, avg_real_acc, avg_fake_acc

            if (epoch+1) % 10 == 0:
                test_conditional_gan(gen, disc, criterion, dict_result, epoch, basedir, test_dataloader, fid)
                torch.save(gen.state_dict(), basedir+f'/saves/model{str(epoch+1)}.pt')

        return dict_result


    def test_conditional_gan(gen, disc, criterion, dict_result, epoch, basedir, test_dataloader, fid):
        gen.eval()
        disc.eval()
        with torch.no_grad():
            epoch_disc_loss, epoch_gen_loss, epoch_real_acc, epoch_fake_acc = [], [], [], []

            for i, (real_imgs, labels) in enumerate(test_dataloader):
                random_labels = torch.randint(0, 10, (real_imgs.shape[0], )).to(device)

                real_imgs = real_imgs.float().cuda()
                noise = torch.randn(LATENT_DIM*real_imgs.shape[0]).to(device)
                fake_imgs = gen(noise.reshape(real_imgs.shape[0], LATENT_DIM))

                disc_real = disc(real_imgs)
                disc_fake = disc(fake_imgs)

                real_acc = torch.sum(torch.round(disc_real) == torch.ones_like(disc_real)) / len(disc_real)
                fake_acc = torch.sum(torch.round(disc_fake) == torch.zeros_like(disc_fake)) / len(disc_fake)
                
                real_loss = criterion(disc_real, torch.ones_like(disc_real))
                fake_loss = criterion(disc_fake, torch.zeros_like(disc_fake))
                disc_loss = (real_loss + fake_loss) / 2

                disc_fake = disc(fake_imgs)
                gen_loss = criterion(disc_fake, torch.ones_like(disc_fake))

                fid.update(fake_imgs, real=False)

                epoch_disc_loss.append(disc_loss.item())
                epoch_gen_loss.append(gen_loss.item())
                epoch_real_acc.append(real_acc.item())
                epoch_fake_acc.append(fake_acc.item())

                progress = int(((i+1)/len(test_dataloader))*20)
                print(f"test [{'='*progress}{' '*(20-progress)}] {i+1}/{len(test_dataloader)}", end='\r', flush=True)

                del disc_loss, gen_loss, real_acc, fake_acc, noise, fake_imgs, disc_real, disc_fake, random_labels

            test_disc_loss = sum(epoch_disc_loss) / len(epoch_disc_loss)
            test_gen_loss = sum(epoch_gen_loss) / len(epoch_gen_loss)
            test_real_acc = sum(epoch_real_acc) / len(epoch_real_acc)
            test_fake_acc = sum(epoch_fake_acc) / len(epoch_fake_acc)
            dict_result['test_disc_loss'].append(test_disc_loss)
            dict_result['test_gen_loss'].append(test_gen_loss)
            dict_result['test_real_acc'].append(test_real_acc)
            dict_result['test_fake_acc'].append(test_fake_acc)
            del epoch_gen_loss, epoch_disc_loss, epoch_real_acc, epoch_fake_acc

            test_graphs(basedir, dict_result)

            fid_score = fid.compute().item()
            fid.reset()
            dict_result['fid_score'].append(fid_score)

            output_imgs(basedir, gen, epoch, test_disc_loss, test_gen_loss, fid_score, labels=True)
            
            gen.train()
            disc.train()
            del fid_score

    train_acgan(disc, gen, disc_optim, gen_optim, disc_criterion, aux_criterion, '.', train_dataloader, test_dataloader)
