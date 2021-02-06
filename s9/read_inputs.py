# Essential Imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import torchvision.transforms as transforms
import torch
from PIL import Image
from albumentations.pytorch import ToTensor
import numpy as np


class TorchImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, is_test=False):
        self.image_list = image_list
        self.is_test = is_test
        self.aug_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.aug_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return (len(self.image_list))

    def __getitem__(self, i):
        image, lebel = self.image_list[i][0], self.image_list[i][1]
        #image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        #image = Image.fromarray(image).convert('RGB')
        if self.is_test == True:
            image = self.aug_test(image)
        else:
            image = self.aug_train(image)
        return image, label


class AlbumentationImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, is_test=False):
        self.image_list = image_list
        self.is_test = is_test
        self.aug_train = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=(-90, 90)),
            A.VerticalFlip(p=0.5),
            A.Blur(blur_limit=16),
            A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ToTensor()
        ])
        self.aug_test = A.Compose([
            A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ToTensor()
        ])

    def __len__(self):
        return (len(self.image_list))

    def __getitem__(self, i):
        image, label = self.image_list[i][0], self.image_list[i][1]
        #image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        #image = Image.fromarray(image).convert('RGB')
        if self.is_test == True:
            image = self.aug_test(image=np.array(image))['image']
        else:
            image = self.aug_train(image=np.array(image))['image']
        return image, label


def read_transform_inputs():
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True)
    trainset = AlbumentationImageDataset(image_list=trainset, is_test=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True)
    testset = AlbumentationImageDataset(image_list=testset, is_test=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return(trainset, trainloader, testset, testloader, classes)


def read_transform_inputs_torch():
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True)
    trainset = TorchImageDataset(image_list=trainset, is_test=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True)
    testset = TorchImageDataset(image_list=testset, is_test=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return(trainset, trainloader, testset, testloader, classes)
