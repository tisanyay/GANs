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

random_seed = 9292
torch.manual_seed(random_seed)
LATENT_DIM = 100
BATCH_SIZE = 128
NUM_EPOCHS = 100
AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS = int(os.cpu_count() / 2)
lr = 3e-4
latent_dims = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
import matplotlib.pyplot as plt

def plot_training_graphs(basedir, disc_loss, gen_loss, real_acc, fake_acc):
    fig = plt.figure(figsize=(15,9))
    plt.plot(disc_loss)
    plt.plot(gen_loss)
    plt.legend(['discriminator loss', 'generator loss'])
    plt.title('Discriminator loss vs Generator loss')
    plt.savefig(basedir+'/graphs/train_loss_graph.jpg')
    plt.close()

    fig = plt.figure(figsize=(15,9))
    plt.plot(real_acc)
    plt.plot(fake_acc)
    plt.legend(['real accuracy', 'fake_accuracy'])
    plt.title('Real accuracy vs fake accuracy')
    plt.savefig(basedir+'/graphs/train_acc_graph.jpg')
    plt.close()

def test_gan(gen, disc, criterion, dict_result, epoch, basedir, test_dataloader, fid):
    gen.eval()
    disc.eval()
    with torch.no_grad():
        epoch_disc_loss, epoch_gen_loss, epoch_real_acc, epoch_fake_acc = [], [], [], []

        for i, (real_imgs, _) in enumerate(test_dataloader):
            # disc_loss, gen_loss, real_acc, fake_acc = dcgan_agent.test_gan(real_imgs)
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

            del disc_loss, gen_loss, real_acc, fake_acc, noise, fake_imgs, disc_real, disc_fake

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

        output_imgs(basedir, gen, epoch, test_disc_loss, test_gen_loss, fid_score)
        
        gen.train()
        disc.train()
        del fid_score

def test_graphs(basedir, dict_result):
    fig = plt.figure(figsize=(15,9))
    plt.plot(dict_result['test_disc_loss'])
    plt.plot(dict_result['test_gen_loss'])
    plt.plot(dict_result['fid_score'])
    plt.legend(['discriminator loss', 'generator loss', 'fid'])
    plt.title('Discriminator loss vs Generator loss vs FID')
    plt.savefig(basedir+'/graphs/test_loss_graph.jpg')
    plt.close()

    fig = plt.figure(figsize=(15,9))
    plt.plot(dict_result['test_real_acc'])
    plt.plot(dict_result['test_fake_acc'])
    plt.legend(['real accuracy', 'fake_accuracy'])
    plt.title('Real accuracy vs fake accuracy')
    plt.savefig(basedir+'/graphs/test_acc_graph.jpg')
    plt.close()

def output_imgs(basedir, gen, epoch, disc_loss, gen_loss, fid, labels=False):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    rows, columns = 8, 6

    if labels:
        random_labels = torch.randint(0, 10, (rows*columns, )).to(device)

    noise = torch.randn(LATENT_DIM*rows*columns).to(device)
    fake_imgs = gen(noise.reshape(rows*columns, LATENT_DIM, 1, 1))
    fig, axs = plt.subplots(rows, columns, figsize=(20, 30))
    for i in range(rows*columns):
        ax = axs[i//columns, i%columns]
        img = (fake_imgs[i].cpu().permute(1,2,0).detach().numpy() + 1) / 2
        ax.imshow(img, cmap='Greys')
        if labels:
            ax.set_title(classes[random_labels[i]])
        ax.axis('off')

    plt.text(-192, -310, f'epoch: {str(epoch)}   discriminator loss: {disc_loss:5.2f}   generator loss: {gen_loss:5.2f}   fid: {fid:5.2f}', fontsize=30, ha='left', va='top')
    plt.savefig(basedir+f'/imgs/img{epoch}.jpg')
    plt.close()

    del rows, columns, noise, fake_imgs

def train_gan(disc, gen, disc_optim, gen_optim, criterion, basedir, train_dataloader, test_dataloader, fid):
    disc.train()
    gen.train()
    train_num_batches = len(train_dataloader)
    dict_result = {}
    dict_result['avg_disc_loss'], dict_result['avg_gen_loss'], dict_result['avg_real_acc'], dict_result['avg_fake_acc'] = [], [], [], []
    dict_result['test_disc_loss'], dict_result['test_gen_loss'], dict_result['test_real_acc'], dict_result['test_fake_acc'], dict_result['fid_score'] = [], [], [], [], []

    test_gan(gen, disc, criterion, dict_result, 0, basedir, test_dataloader, fid)
    
    for epoch in range(NUM_EPOCHS):
        epoch_disc_loss, epoch_gen_loss, epoch_real_acc, epoch_fake_acc = [], [], [], []
        for i, (real_imgs, _) in enumerate(train_dataloader):
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
            disc.zero_grad()
            disc_loss.backward(retain_graph=True)
            disc_optim.step()

            epoch_disc_loss.append(disc_loss.item()) 
            epoch_real_acc.append(real_acc.item()) 
            epoch_fake_acc.append(fake_acc.item()) 

            disc_fake = disc(fake_imgs)
            gen_loss = criterion(disc_fake, torch.ones_like(disc_fake))
            gen.zero_grad()
            gen_loss.backward()
            gen_optim.step()

            epoch_gen_loss.append(gen_loss.item())

            progress = int(((i+1)/train_num_batches)*20)
            print(f"{epoch+1}/{NUM_EPOCHS} [{'='*progress}{' '*(20-progress)}] {i+1}/{train_num_batches}", end='\r', flush=True)

            del disc_loss, gen_loss, real_acc, fake_acc, noise, fake_imgs, disc_real, disc_fake, real_loss, fake_loss

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

        print(f'disc loss: {avg_disc_loss:5.2f}  gen loss: {avg_gen_loss:5.2f}  real acc: {avg_real_acc:5.2f}  fake_acc: {avg_fake_acc:5.2f}')
        del avg_disc_loss, avg_gen_loss, avg_real_acc, avg_fake_acc

        if (epoch+1) % 10 == 0:
            test_gan(gen, disc, criterion, dict_result, epoch, basedir, test_dataloader, fid)
            torch.save(gen.state_dict(), basedir+f'/saves/model{str(epoch+1)}.pt')

    return dict_result

# compute the current classification accuracy
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)