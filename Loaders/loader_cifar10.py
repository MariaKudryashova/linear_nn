from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

class loader():
    
    def __init__(self):
        super(loader, self).__init__()  
        #Define transformations for the training set, flip the images randomly, crop out and apply mean and std normalization
        self.train_transformations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32,padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        root = '../Loaders/'
        #Load the training set
        self.train_set =CIFAR10(root=root,train=True,transform=self.train_transformations,download=True)

        #Create a loder for the training set
        self.train_loader = DataLoader(self.train_set,batch_size=32,shuffle=True,num_workers=4)


        # Define transformations for the test set
        self.test_transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        ])

        # Load the test set, note that train is set to False
        self.test_set = CIFAR10(root=root, train=False, transform=self.test_transformations, download=True)

        # Create a loder for the test set, note that both shuffle is set to false for the test loader
        self.test_loader = DataLoader(self.test_set, batch_size=32, shuffle=False, num_workers=4)
