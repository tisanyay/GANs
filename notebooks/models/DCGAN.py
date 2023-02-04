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
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1) # 16x16

        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # 8x8
        self.bn1 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) # 4x4
        self.bn2 = nn.BatchNorm2d(512)

        self.conv4 = nn.Conv2d(512, 1, kernel_size=4, stride=4, padding=0)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.bn1(F.leaky_relu(self.conv2(x), 0.2))
        x = self.bn2(F.leaky_relu(self.conv3(x), 0.2))
        x = self.conv4(x)
        return torch.sigmoid(x)

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(latent_dim, 1024, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(1024)

        self.conv2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        
        self.conv3 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = x.view(-1, 100, 1, 1)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.conv4(x)
        return torch.tanh(x)