import torch as torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DataRead import OurData

from kan import KAN, create_dataset
#literally copied the code below from the documentation.  Need to figure out what does what

# Set the default data type to double
torch.set_default_dtype(torch.float64)
#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 28*28 # 28*28

'''
initialize model
- refer to MultKAN.py for more information
    width: number of neurons in each layer, in order from input to output
    k: order of the spline
    seed: random seed
    grid: grid intervals/grid points (affects the accuracy of the splines/learnable functions)
'''
model = KAN(width=[2, 5, 1], grid=5, k=3, seed=0, device=device)
model2 = KAN(width=[2, 5, 1], grid=5, k=3, seed=0, device=device)

#to-do: modify f into a function that returns the values from out dataset

#Our dataset
data = OurData()
ourdata = {}
for key in data.ourdataset:
    ourdata[key] = data.ourdataset[key]

model(ourdata['train_input']) #forward pass of the model
model.plot() #plots the model
print("done")
'''
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2, device=device)
model2(dataset['train_input'])
model2.plot()
print(dataset['train_input'], ourdata['train_input'])
'''

#code to train the model
'''
Training the model off the dataset
- opt: optimization method (LBFGS)
- steps: training steps
- lamb: penalty parameter
other parameters: lr = learning rate = 1, loss_fn = loss function = None
'''
#fits the model to the dataset
'''
model.fit(ourdata, opt="LBFGS", steps=50, lamb=0.001) #values from the basic example in the documentation
model.plot() #plots the model
print("done")

model = model.prune()
model.plot()
'''
