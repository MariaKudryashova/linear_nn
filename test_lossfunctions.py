#Простая полносвязная нейронная сеть

import sys
sys.path.append('./Loaders')
sys.path.append('../')

import torch
import torch.nn as nn

import torchsummary
import seaborn as sns
from matplotlib import pyplot as plt

import train_loop
import LoaderMnist
import Saver

#Constants
features = 784
classes = 10
batch_size = 128
max_epochs = 10
is_break = False
is_linet = False

device = "cuda" if torch.cuda.is_available() else "cpu"
loader = LoaderMnist.Loader(batch_size, (0.5), (0.5))

name_title = "Lf_of_FC"
Saver = Saver.Saver(name_title, "./Comparisons/")

array_criterions = {"CrossEntropyLoss": nn.CrossEntropyLoss(), 
                    "MSELoss": nn.MSELoss(), 
                    "TripletMarginLoss": nn.TripletMarginLoss(),
                    "SoftMarginLoss":nn.SoftMarginLoss}

accuracy_valids = {}

model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features, 128),
            nn.ReLU(),
            nn.Linear(128, classes)
        )   

model.to(device)

for i, k in enumerate(array_criterions.keys()):
    
    criterion = array_criterions[k] 
    optimizer = torch.optim.Adam(model.parameters())

    # torchsummary.summary(model, (features,), batch_size=batch_size)

    #обучение модели
    accuracy = train_loop.train_loop(device, max_epochs, 
    model, criterion, optimizer, loader.loaders, is_linet, is_break)

    print(f"{i+1} of {len(array_criterions)}  accuracy {k}: ", accuracy["valid"])

    accuracy_valids[k] = accuracy["valid"]


sns.set(style="darkgrid", font_scale=1.4)

fig = plt.figure(figsize=(16, 10))
plt.title("Valid accuracy")
plt.plot(range(max_epochs), accuracy_valids["CrossEntropyLoss"], label="CrossEntropyLoss", linewidth=2)
plt.plot(range(max_epochs), accuracy_valids["MSELoss"], label="MSELoss", linewidth=2)
plt.plot(range(max_epochs), accuracy_valids["TripletMarginLoss"], label="TripletMarginLoss", linewidth=2)
plt.plot(range(max_epochs), accuracy_valids["SoftMarginLoss"], label="SoftMarginLoss", linewidth=2)
plt.legend()
plt.xlabel("Epoch")
plt.show()

fig.savefig(f"Comparisons\{name_title}.png")

Saver.SaveAccuracy(accuracy_valids)
