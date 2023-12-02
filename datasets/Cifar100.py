import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
from PIL import Image
import glob

class Cifar10(CIFAR100):

    def __init__(self, root='/', train=True, transform=None):
        super().__init__(root=root, train=train, transform=transform, download=True)
        self.num_classes = 100

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        return super().__getitem__(index)