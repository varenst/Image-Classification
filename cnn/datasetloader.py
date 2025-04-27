import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_data(data_dir="./data"):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    batch_size = 4

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    trainset, valset = random_split(full_dataset, [train_size, val_size])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = full_dataset.classes

    return trainloader, valloader, classes
