from medmnist import PneumoniaMNIST
from torch import *
import pandas
import numpy


trainingData = PneumoniaMNIST(split="train",download=True,size=28)
validationData =  PneumoniaMNIST(split="val",download=True,size=28)
testData = PneumoniaMNIST(split="test",download=True,size=28)

