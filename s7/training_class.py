import torch
import torch.nn as nn
import torch.nn.functional as F
from truth_checker import truth_checker


class training_class:
    def __init__(self, model, trainloader, testloader, device, epoch, optimizer):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.epoch = epoch
        self.optimizer = optimizer
        self.loss = 0

    def fit(self):
        criterion = nn.CrossEntropyLoss()
        for x_epoch in range(self.epoch):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                # Loss

                self.loss = criterion(outputs, labels)
                self.loss.backward()
                self.optimizer.step()
            # Epoch Loss
            test_check = truth_checker(
                self.model, self.testloader, self.device)
            #train_check = truth_checker(self.model,self.trainloader)
            print('epoch [%d] train accuracy %s : test accuracy %.3f' %
                  (x_epoch, 'train - NA', test_check))
