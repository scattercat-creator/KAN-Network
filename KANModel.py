import torch as torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DataRead import OurData
import traceback

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
model = KAN(width=[784, 5, 1], grid=5, k=3, seed=0, device=device)
#model2 = KAN(width=[2, 5, 1], grid=5, k=3, seed=0, device=device)

#to-do: modify f into a function that returns the values from out dataset

#Our dataset
data = OurData()
ourdata = {}
for key in data.ourdataset:
    ourdata[key] = (data.ourdataset[key])[:10] #only get the first 10 data points for now

 #check the shape of our dataset against the create_dataset function which we know works
f = lambda x: x[:, [0]] + x[:, [1]] # This function takes a 2D tensor and returns the sum of the first and second columns
dataset = create_dataset(f, n_var=2, device=device)
'''
print(dataset['train_input'].shape, ourdata['train_input'].shape)
print(dataset['train_label'].shape, ourdata['train_label'].shape)
    #checking to see if the data points are only 0s(it looks like it from the outputs but they are not)
print(ourdata['train_input'][1])
print(ourdata['train_input'][2])
'''
ourdata['train_input'] = ourdata['train_input'] + 1 #add 1 to the data points to make them non-zero. Didnt fix the coef error
model(ourdata['train_input']) #forward pass of the model
#model.plot() #plots the model, avoid doing this since it will plot functions for all the neurons(and we have a lot since we are dealing with images)
#code to train the model
'''
Training the model off the dataset
- opt: optimization method (LBFGS)
- steps: training steps
- lamb: penalty parameter
other parameters: lr = learning rate = 1, loss_fn = loss function = None
'''
#fits the model to the dataset
try:
    #currently doesn't work since all the coefficients are uninitialized???
    model.fit(ourdata, opt="LBFGS", steps=50, lamb=0.001,) #values from the basic example in the documentation
#model.plot() 
except Exception as e:
    print(e)
    traceback.print_exc()
print("done")
'''
model = model.prune()
model.plot()
'''
