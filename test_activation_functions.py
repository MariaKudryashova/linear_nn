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
name_title = "Af_of_FC"
Saver = Saver.Saver(name_title, "./Comparisons/")

array_activations = {"None": None, 
                    "ELU": nn.ELU(), 
                    "ReLU": nn.ReLU(), 
                    "LeakyReLU": nn.LeakyReLU()}

accuracy_valids = {}

for k in enumerate(array_activations.keys()):
    
    if (array_activations[k]):
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features, 128),
            array_activations[k],
            nn.Linear(128, classes)
        )
    else:
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features, 128),
            nn.Linear(128, classes)
        )

    model.to(device)

    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters())

    # torchsummary.summary(model, (features,), batch_size=batch_size)

    #обучение модели
    accuracy = train_loop.train_loop(device, max_epochs, 
    model, criterion, optimizer, loader.loaders, is_linet, is_break)

    #print(f"accuracy {k}: ", accuracy["valid"])
    print(f"{i+1} of {len(array_criterions)}  accuracy {k}: ", accuracy["valid"])

    accuracy_valids[k] = accuracy["valid"]
    # break


sns.set(style="darkgrid", font_scale=1.4)

fig = plt.figure(figsize=(16, 10))
plt.title("Valid accuracy")
plt.plot(range(max_epochs), accuracy_valids["None"], label="No activation", linewidth=2)
plt.plot(range(max_epochs), accuracy_valids["ELU"], label="ELU activation", linewidth=2)
plt.plot(range(max_epochs), accuracy_valids["ReLU"], label="ReLU activation", linewidth=2)
plt.plot(range(max_epochs), accuracy_valids["LeakyReLU"], label="LeakyReLU activation", linewidth=2)
plt.legend()
plt.xlabel("Epoch")
#plt.show()
fig.savefig(f"Comparisons\{name_title}.png")

Saver.SaveAccuracy(accuracy_valids)
