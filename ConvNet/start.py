from torchvision.datasets import MNIST
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from torch import nn
import torch
from ConvNetModule import ConvNet  

num_epochs = 3 
num_classes = 10 
batch_size = 100 
learning_rate = 0.001

model = ConvNet(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

data_tfs = tfs.Compose([tfs.ToTensor()])
root = "../Loaders"
train_dataset = MNIST(root, train=True,  transform=data_tfs, download=True)
val_dataset  = MNIST(root, train=False, transform=data_tfs, download=True)
train = MNIST(root, train=True,  transform=data_tfs, download=True)
valid = MNIST(root, train=False, transform=data_tfs, download=True)
train_loader = DataLoader(train, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(valid, batch_size=batch_size, drop_last=True)


total_step = len(train_loader)
loss_list = []
acc_list = []


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if (i > 5): break
        # Прямой запуск
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Обратное распространение и оптимизатор
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Отслеживание точности
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)
        
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))


model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

# Сохраняем модель и строим график
torch.save(model.state_dict(), '../Models/convnet_model.ckpt')