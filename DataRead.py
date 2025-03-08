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

class OurData:
    def __init__(self):
        self.ourdataset = {}
        train_images = read_idx('/workspaces/KAN-Network/Dataset/train-images-idx3-ubyte/train-images-idx3-ubyte') #contains the training data, each data is the binary representation of an image as per the MNIST dataset
        self.ourdataset['train_input'] = (train_images).view(-1, 28*28) #reshapes the data into a 2D array, each row is an image
        test_images = read_idx('/workspaces/KAN-Network/Dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte') #contains the testing data, same format as train_input
        self.ourdataset['test_input'] = (test_images).view(-1, 28*28) #reshapes the data into a 2D array, each row is an image
        self.ourdataset['train_label'] = read_idx('/workspaces/KAN-Network/Dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte').unsqueeze(1) #contains the labels for the training data
        self.ourdataset['test_label'] = read_idx('/workspaces/KAN-Network/Dataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte').unsqueeze(1) #contains the labels for the testing data
        #print("These are our training inputs and labels")
    def __getitem__(self):
        return self.ourdataset
    def __getitem__(self, key):
        return self.ourdataset[key]