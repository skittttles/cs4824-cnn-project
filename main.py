from medmnist import PneumoniaMNIST

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

trainingData = PneumoniaMNIST(split = "train", download = True, size = 28, transform = data_transform)
validationData =  PneumoniaMNIST(split = "val", download = True,size = 28, transform = data_transform)
testData = PneumoniaMNIST(split = "test", download = True, size = 28, transform = data_transform)

train_loader = data.DataLoader(trainingData, batch_size = 64, shuffle = True)
val_loader = data.DataLoader(validationData, batch_size = 64, shuffle = False)
test_loader = data.DataLoader(testData, batch_size = 64, shuffle = False)

print(trainingData)
print("===================")
print(testData)

trainingData.montage(length=1).save("montage.png")
