from medmnist import PneumoniaMNIST
from torch import *
import pandas
import numpy


dataset = PneumoniaMNIST(split="val",download=False,size=64)

print(dataset)