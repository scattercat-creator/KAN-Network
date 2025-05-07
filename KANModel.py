import torch as torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DataRead import OurData
import traceback
import struct
import matplotlib.colors as mcolors

from Visualization import DataVisual
from DataRead import OurData, compress
from kan import KAN

torch.set_default_dtype(torch.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 28*28 # from 28*28

#total train datapoints=60000, test datapoints=10000.
data = OurData()
#edit the parameters if you are loading a pretrained model so you feed it new data. Or you can train it again with the same data
ourdata = data.filldata(50, 784)
ourdata2 = data.filldata(500, 112)
ourdata3 = data.filldata(2500, 28)
#print(ourdata['train_label'].size(), ourdata2['train_label'].size(), ourdata3['train_label'].size())
#print(ourdata['train_input'].size(), ourdata2['train_input'].size(), ourdata3['train_input'].size())

'''
initialize model
- refer to MultKAN.py for more information
    width: number of neurons in each layer, in order from input to output
    k: order of the spline
    seed: random seed
    grid: grid intervals/grid points (affects the accuracy of the splines/learnable functions)
'''
model = KAN(width=[784, 80, 10], grid=5, k=3, seed=0, device=device)
model2 = KAN(width=[196, 40, 10], grid=5, k=3, seed=0, device=device, ckpt_path='./model2')
model3 = KAN(width=[28, 20, 10], grid=5, k=3, seed=0, device=device, ckpt_path='./model3')

'''to load a model from a checkpoint, uncomment the following code
model = model.loadckpt('./model')
model2 = model2.loadckpt('./model2')
model3 = model3.loadckpt('./model3')
'''
'''if you are creating a new model from scratch, uncomment the following code
model(ourdata['train_input']) #forward pass of the model to initialize the splines
model2(ourdata2['train_input']) 
model3(ourdata3['train_input']) 
#model.plot() #plots the model, avoid doing this since it will plot functions for all the neurons(and we have a lot since we are dealing with images)
'''

''' code to now fit the model to the dataset

Training the model off the dataset
- opt: optimization method (LBFGS)
- steps: training steps
- lamb: penalty parameter
other parameters: lr = learning rate = 1, loss_fn = loss function = None
'''
#steps = intervals to divide the dataset and update model, epochs = how many times the entire dataset is passed through the model
modelresults = []
modelresults.append(model.fit(ourdata, opt="LBFGS", steps=20, lamb=0.001))
modelresults.append(model2.fit(ourdata2, opt="LBFGS", steps=40, lamb=0.001))
modelresults.append(model3.fit(ourdata3, opt="LBFGS", steps=80, lamb=0.001))
'''
try:
    
except Exception as e:
    print(e)
    traceback.print_exc()'
'''

eval1 = model.evaluate(ourdata)
eval2 = model2.evaluate(ourdata2)
eval3 = model3.evaluate(ourdata3)
evaluation_results = [eval1, eval2, eval3]
test_losses = [[result['test_loss'] for result in evaluation_results]]

testingdata = data.getitems(9950, 10000) 
predictions = model.forward(testingdata[0])
predictions2 = model2.forward(compress(testingdata[0], 196))
predictions3 = model3.forward(compress(testingdata[0], 28))
allpredictions = [predictions, predictions2, predictions3]

pred1 = DataVisual(testingdata[1], predictions)
pred2 = DataVisual(testingdata[1], predictions2)
pred3 = DataVisual(testingdata[1], predictions3)

model.saveckpt(path='./model_checkpoint') #saves the model into the files
model2.saveckpt(path='./model2_checkpoint')
model3.saveckpt(path='./model3_checkpoint')
''' load/save models?
loaded_model = model.loadckpt(path='./model/0.1')
loaded_model2 = model2.loadckpt(path='./model2/0.1')
loaded_model3 = model3.loadckpt(path='./model3/0.1')'
'''


''' accuracy? wip. Will probably relocate to Visualization.py instead
error = [0, 0, 0]
correct = [0, 0, 0]
for i in range(len(testingdata[1])):
    print(i, end="\t")
    for j in range(3):
        print(j+1, allpredictions[j][i].item(), end="\t")
        difference = abs(allpredictions[j][i].item()-i)
        error[j] += 1 if difference > 1 else difference
    print()
for j in range(3):
    error[j] /= len(testingdata[1])
    correct[j] = 100 - 100*error[j]

plt.scatter([i for i in range(1, 4)], error, label='Average Error') 
plt.scatter([i for i in range(1, 4)], correct, label='Accuracy') 
plt.xlabel('Model Number')
plt.title('Scatter Plot of Models')
plt.legend()
print(error)
print(correct)
'''
#loaded_model.loadckpt(path='./model_checkpoint') #loads the model from the files