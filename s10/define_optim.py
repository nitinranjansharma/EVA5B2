import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


def optim_define(model):
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    return(optimizer)
