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
NUM_EPOCHS = 500
AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS = int(os.cpu_count() / 2)
lr = 3e-4
latent_dims = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

class ConditionalDCGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(10, 256)

        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1) # 16x16
        self.conv2 = nn.Conv2d(129, 256, kernel_size=4, stride=2, padding=1) # 8x8
        self.conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) # 4x4
        self.conv4 = nn.Conv2d(512, 1, kernel_size=4, stride=4, padding=0)

    def forward(self, x, labels):
        y = F.one_hot(labels, num_classes=10).view(-1, 10).float()
        y = F.leaky_relu(self.lin1(y), 0.2).view(-1, 1, 16, 16)

        x = F.leaky_relu(self.conv1(x), 0.2)

        x = torch.concat([x,y], 1)

        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.conv4(x)
        return torch.sigmoid(x)

class ConditionalDCGANGenerator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1_x = nn.ConvTranspose2d(latent_dim, 1024, kernel_size=4, stride=2, padding=0) # 4x4
        self.bn1_x = nn.BatchNorm2d(1024)

        self.conv1_y = nn.ConvTranspose2d(10, 1024, kernel_size=4, stride=2, padding=0) # 4x4
        self.bn1_y = nn.BatchNorm2d(1024)

        self.conv2 = nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1) # 8x8
        self.bn2 = nn.BatchNorm2d(1024)

        self.conv3 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1) # 16x16
        self.bn3 = nn.BatchNorm2d(512)

        self.conv4 = nn.ConvTranspose2d(512, 3, kernel_size=4, stride=2, padding=1) # 32x32

    def forward(self, x, labels):
        x = x.view(-1, 100, 1, 1)
        x = F.relu(self.bn1_x(self.conv1_x(x)))

        y = F.one_hot(labels, num_classes=10).view(-1, 10, 1, 1).float()
        y = F.relu(self.bn1_y(self.conv1_y(y)))

        x = torch.concat([x, y], 1)

        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.conv4(x)
        return torch.tanh(x)