import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, num_classes): 
        super(ConvNet, self).__init__() 
        self.layer1 = nn.Sequential( 
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(16),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.layer2 = nn.Sequential( 
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)) 
        #self.drop_out = nn.Dropout() 
        self.fc1 = nn.Linear(7 * 7 * 32, 1000) 
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, num_classes)

    def forward(self, x): 
        out = self.layer1(x) 
        out = self.layer2(out) 
        out = out.reshape(out.size(0), -1) 
        #out = self.drop_out(out) 
        out = self.fc1(out) 
        out = self.fc2(out) 
        out = self.fc3(out) 
        return out