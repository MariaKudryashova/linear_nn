import sys
sys.path.append('../Loaders/')

import LoaderMnist

import torch
import torch.nn as nn

import matplotlib.pyplot as plt 
import numpy as np
from torch.autograd import Variable

#загрузка модели
name = "../Models/model_FC_ReLU_CE_20230323.pt"
batch_size = 128

loader = LoaderMnist.Loader(batch_size, (0.5), (0.5))

model = torch.jit.load(name)
model.eval()

img, target = loader.get_img()
print(target)

img_tensor = loader.image_transform(img).float()
img_tensor = img_tensor.unsqueeze_(0)
input = Variable(img_tensor)
output = model(input)

# print(output.data.numpy())

index = output.data.numpy().argmax()
print(index)



