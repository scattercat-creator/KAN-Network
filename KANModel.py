import torch as torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DataRead import OurData
import traceback

from kan import KAN, create_dataset
#literally copied the code below from the documentation.  Need to figure out what does what

# Set the default data type to double
torch.set_default_dtype(torch.float32)
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
model = KAN(width=[784, 10, 1], grid=5, k=3, seed=0, device=device)
model2 = KAN(width=[28, 10, 1], grid=5, k=3, seed=0, device=device, ckpt_path='./model2')
model3 = KAN(width=[3, 5, 1], grid=5, k=3, seed=0, device=device, ckpt_path='./model3')

data = OurData() #Our dataset
ourdata = {}
ourdata2 = {}
ourdata3 = {}
for key in data.ourdataset:
    #total train datapoints=60000, test datapoints=10000. approx time:
    ourdata[key] = data.ourdataset[key][:100] #only get the first 100 data points for now
    ourdata2[key] = data.getsliced(key, 28)[:1000] #each element is a sum of a row of the 2d array
    ourdata3[key] = data.getsliced(key, 3)[:2000] #each element is a sum of a third of the entire 2d array

#this is where the error is originating from. coef is uninitialized in the forward pass due to a matrix opteration error, possibly due to our differences in our datasets
''' inside the initial curve2coef function
try:
    coef = torch.linalg.lstsq(mat, y_eval).solution[:,:,:,0]
except Exception as e:
    print('lstsq failed:', e)

keeps returning the exception for our dataset but not for the dataset created by create_dataset
'''

model(ourdata['train_input']) #forward pass of the model
model2(ourdata2['train_input']) 
model3(ourdata3['train_input']) 
#print("model pass complete")
#model.plot() #plots the model, avoid doing this since it will plot functions for all the neurons(and we have a lot since we are dealing with images)

''' code to now fit the model to the dataset
Training the model off the dataset
- opt: optimization method (LBFGS)
- steps: training steps
- lamb: penalty parameter
other parameters: lr = learning rate = 1, loss_fn = loss function = None
'''
modelresults = []
modelresults.append(model.fit(ourdata, opt="LBFGS", steps=50, lamb=0.001)) #values from the basic example in the documentation
modelresults.append(model2.fit(ourdata2, opt="LBFGS", steps=50, lamb=0.001))
modelresults.append(model3.fit(ourdata3, opt="LBFGS", steps=50, lamb=0.001))
#model.plot() #plots the model, avoid doing this since it will plot functions for all the neurons(and we have a lot since we are dealing with images)

eval1 = model.evaluate(ourdata)
eval2 = model2.evaluate(ourdata2)
eval3 = model3.evaluate(ourdata3)
evaluation_results = [eval1, eval2, eval3]
test_losses = [[result['test_loss'] for result in evaluation_results]]

testingdata = data.getitems(9950, 10000) 
results = model.forward(testingdata[0])
results2 = model2.forward(testingdata[0])
results3 = model3.forward(testingdata[0])

