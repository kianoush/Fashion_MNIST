"""
Fashion_MNIST
"""
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2


"""
Variable
"""
batch_size = 32


"""
Load data
"""
train_data = torchvision.datasets.FashionMNIST('./datasets', train=True, transform=transforms.ToTensor(), download=True)
test_data = torchvision.datasets.FashionMNIST('./datasets', train=False, transform=transforms.ToTensor(), download=True)

train_dl = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


print('End!')