# Общий модуль для загрузки стандартных датасетов

from torchvision.datasets import MNIST
import torchvision.transforms as tfs
from statistics import mean
import torchsummary
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class Loader():

    def __init__(self, _batch_size, _norm_a, _norm_b):
        super(Loader, self).__init__()  

        self.image_transform = tfs.Compose([
                            tfs.ToTensor(),
                            tfs.Normalize(_norm_a, _norm_b)
                        ])

        #(X-mean)/std 
        # self.invert_normalize = tfs.Compose([
        #             tfs.Normalize([-1], [2]),
        # ])
        
        root = '../Loaders/'
        
        self.train = MNIST(root, train=True,  transform=self.image_transform, download=True)
        self.valid = MNIST(root, train=False, transform=self.image_transform, download=True)
            
        self.train_dataloader = DataLoader(self.train, batch_size=_batch_size, drop_last=True)
        self.valid_dataloader = DataLoader(self.valid, batch_size=_batch_size, drop_last=True)

        self.mnist_samples = MNIST(root, train=False, download=True, transform=None)
        
        self.loaders = {"train": self.train_dataloader, 
                            "valid": self.valid_dataloader}


    def get_img(self):
        i = np.random.randint(low=0, high=10000)
        img, target = self.mnist_samples[i]       
        # plt.imshow(img, cmap="gray")
        # plt.show()
        
        return img, target 

    
    