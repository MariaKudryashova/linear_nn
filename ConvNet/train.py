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


# # Train the model
# accuracy = {"train": [], "valid": []}

# for epoch in range(max_epochs):
#     for i, (images, labels) in enumerate(loader.loaders["train"]):
#         images = images.to(device)
#         labels = labels.to(device)
        
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         accuracy["train"].append(loss.item)
        
#         if (i+1) % 100 == 0:
#             print ('Epoch {}, Loss: {:.4f}' 
#                    .format(epoch+1, loss.item()))  #Step [{}/{}]



# model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in enumerate(loader.loaders["train"]):
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#         acc = correct / total
#         accuracy["valid"].append(acc)

#     print('Test Accuracy of the model on the 10000 test images: {} %'.format(acc * 100))

# # Save the model checkpoint
# # torch.save(model.state_dict(), 'model.ckpt')

# print("accuracy: ", accuracy)

# #сохранение модели
# saver.Save(model, accuracy)
