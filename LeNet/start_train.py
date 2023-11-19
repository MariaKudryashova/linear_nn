import LeNet

import sys
sys.path.append('../Loaders')
sys.path.append('../')

import train_loop
import LoaderMnist

import torch
import torch.nn as nn

import matplotlib.pyplot as plt 
import numpy as np

import Saver

#CONSTANTS
num_classes = 10 
max_epochs = 5
batch_size = 128
is_break = False


Saver = Saver.Saver("LeNet5_CE", "../Models")
device = "cuda" if torch.cuda.is_available() else "cpu"
loader = LoaderMnist.Loader(batch_size, (0.5), (0.5))
# model = LeNet.LeNet("LeNet", num_classes).to(device)
model = LeNet.LeNet5("LeNet5", num_classes).to(device)

criterion = nn.CrossEntropyLoss() #nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# torchsummary.summary(model, (features,), batch_size=batch_size)

#обучение модели
accuracy = train_loop.train_loop(device, max_epochs, 
model, criterion, optimizer, loader.loaders, False, is_break)

print("accuracy: ", accuracy["valid"])

#сохранение модели
Saver.SaveAll(model, accuracy)
