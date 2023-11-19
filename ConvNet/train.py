import ConvNet

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
max_epochs = 4
batch_size = 100
is_break = False
learning_rate = 0.001

Saver = Saver.Saver("ConvNet_CE", "../Models")

device = "cuda" if torch.cuda.is_available() else "cpu"
loader = LoaderMnist.Loader(batch_size, (1), (1))

model = ConvNet.ConvNet(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

# torchsummary.summary(model, (features,), batch_size=batch_size)

#обучение модели
accuracy = train_loop.train_loop(device, max_epochs, 
model, criterion, optimizer, loader.loaders, False, is_break)

print("accuracy: ", accuracy["valid"])

#сохранение модели
Saver.SaveAll(model, accuracy)
