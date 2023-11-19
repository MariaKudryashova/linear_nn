#Простая полносвязная нейронная сеть
import sys
sys.path.append('../Loaders')
sys.path.append('../')

import torch
import torch.nn as nn

import torchsummary

import train_loop
import LoaderMnist
import Saver

#Constants
features = 784
classes = 10
batch_size = 128
max_epochs = 10
is_break = False

device = "cuda" if torch.cuda.is_available() else "cpu"
loader = LoaderMnist.Loader(batch_size, (0.5), (0.5))
Saver = Saver.Saver("FC_ReLU_CE", "../Models")

#Activation function
activation = nn.ReLU()

#Set up a model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(features, 128),
    activation,
    nn.Linear(128, classes)
)
model.to(device)

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters())

torchsummary.summary(model, (features,), batch_size=batch_size)

#обучение модели
accuracy = train_loop.train_loop(device, max_epochs, 
model, criterion, optimizer, loader.loaders, True, is_break)

print("accuracy: ", accuracy["valid"])

Saver.SaveAll(model, accuracy)

