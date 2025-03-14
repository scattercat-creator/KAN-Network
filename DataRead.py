import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import struct

torch.set_default_dtype(torch.float32)
    
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, dtype, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        return torch.tensor(np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)).to(torch.float32)
    
def compress(dataset, slice = 1):
        fraction = int(784/slice)
        arr = torch.tensor([])
        for i in range(slice):
            arr = torch.cat((arr, dataset[:, i*fraction:(i+1)*fraction].sum(dim=1, keepdim=True)), dim=1)
        return arr

class OurData:
    def __init__(self):
        self.ourdataset = {}
        train_images = read_idx('/workspaces/KAN-Network/Dataset/train-images-idx3-ubyte/train-images-idx3-ubyte') #contains the training data, each data is the binary representation of an image as per the MNIST dataset
        self.ourdataset['train_input'] = (train_images).view(-1, 28*28) #reshapes the data into a 2D array, each row is an image
        test_images = read_idx('/workspaces/KAN-Network/Dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte') #contains the testing data, same format as train_input
        self.ourdataset['test_input'] = (test_images).view(-1, 28*28) #reshapes the data into a 2D array, each row is an image
        
        #to-do: convert the lablels into 10 element arrays for classification
        train_label = read_idx('/workspaces/KAN-Network/Dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte').unsqueeze(1) #contains the labels for the training data
        test_label = read_idx('/workspaces/KAN-Network/Dataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte').unsqueeze(1) #contains the labels for the testing data
        self.ourdataset['train_label'] = torch.zeros(len(train_label), 10)#, 1)
        self.ourdataset['test_label'] = torch.zeros(len(test_label), 10)#, 1)
        #print(self.ourdataset['train_label'].size(), self.ourdataset['test_label'].size())
        #the code below assigns a value of 1 to the correct labels in the 10 element array, everything else is 0
        for i in range(len(train_label)):
            self.ourdataset['train_label'][i][train_label[i].long()] = 1
        for i in range(len(test_label)):
            self.ourdataset['test_label'][i][test_label[i].long()] = 1
        #print("These are our training inputs and labels")
        #self.ourdataset['train_label'] = self.ourdataset['train_label'].view(-1, 10)
        #self.ourdataset['test_label'] = self.ourdataset['test_label'].view(-1, 10)
    def __getitem__(self):
        return self.ourdataset
    def getitems(self, index, endindex = 10000, test = True):
        key = 'test_input' if test else 'train_input'
        key2 = 'test_label' if test else 'train_label'
        return [self.ourdataset[key][index:endindex], self.ourdataset[key2][index:endindex]]
    def filldata(self, m_ins, m_slices, start=0):
        ourdatas = {}
        for key in self.ourdataset:
            isin = (key == 'train_input' or key == 'test_input')
            thisdata = self.ourdataset[key][start:start+m_ins]
            ourdatas[key] = (thisdata if (m_slices == 784 or not isin) else compress(thisdata, m_slices))
        return ourdatas

