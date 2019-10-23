"""
Fashion_MNIST
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import cv2

use_GPU = torch.cuda.is_available()

print('Cuda', use_GPU)


"""
Variable
"""
batch_size = 32


"""
Load data
"""
train_datasets = torchvision.datasets.FashionMNIST('./datasets', train=True, transform=transforms.ToTensor(), download=True)
test_datasets = torchvision.datasets.FashionMNIST('./datasets', train=False, transform=transforms.ToTensor(), download=True)

train_dl = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(dataset=test_datasets, batch_size=batch_size, shuffle=True)



"""
Modle
"""
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Sequential(
                                    nn.Conv2d(1, 16, 3, 1, 2),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2)
                                     )
        self.layer2 = nn.Sequential(
                                    nn.Conv2d(16, 32, 3, 1, 2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2)
                                     )
        self.fc = nn.Linear(8*8*32, 10)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2 = out2.view(out2.size(0), -1)
        y = self.fc(out2)
        return y


model = SimpleModel()

if use_GPU:
    model = model.cuda()

"""
Loss
"""
criterions = nn.CrossEntropyLoss()

"""
Optim
"""
optim = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


"""
Main loop
"""

def to_var(x, volatile=False):
    if use_GPU:
        x = x.cuda()
    return Variable(x, volatile=volatile)

num_epochs = 10
losses = []
for epoch in range(num_epochs):
    model.train()
    for i,(inputs, targets) in enumerate(train_dl):

        inputs = to_var(inputs)
        targets = to_var(targets)

        # forward
        optim.zero_grad()
        output = model(inputs)

        # Loss
        loss = criterions(output, targets)
        losses += [loss.data.item()]

        # backward
        loss.backward()

        # update parameters
        optim.step()

        if (i+1) % 100==0:
            print('Train, Epoch [%2d/%2d], Step [%3d/%3d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dl), loss.data.item()))




print('End!')