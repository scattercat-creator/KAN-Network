import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def read_idx_images(file_path):
    """ Reads an IDX image file and returns a tensor of shape (N, 28, 28) """
    with open(file_path, 'rb') as f:
        f.read(4)  # Skip magic number
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        images = torch.tensor(data, dtype=torch.float32)  # Convert to float

        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(20), #change the size of the image to 20x20
        transforms.ToTensor()
        ])
        resized_images = torch.stack([transform(image) for image in images])
        return resized_images

def read_idx_labels(file_path):
    """ Reads an IDX label file and returns a tensor of shape (N,) """
    with open(file_path, 'rb') as f:
        f.read(4)  # Skip magic number
        num_labels = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return torch.tensor(data, dtype=torch.long)  # Convert to long tensor

class OurData:
    def __init__(self):
        self.ourdataset = {}
        self.ourdataset['train_input'] = read_idx_images('/workspaces/KAN-Network/Dataset/t10k-images.idx3-ubyte') #contains the training data, each data is the binary representation of an image as per the MNIST dataset
        self.ourdataset['test_input'] = read_idx_images('/workspaces/KAN-Network/Dataset/t10k-images.idx3-ubyte') #contains the testing data, same format as train_input
        self.ourdataset['train_label'] = read_idx_labels('/workspaces/KAN-Network/Dataset/train-labels.idx1-ubyte') #contains the labels for the training data
        self.ourdataset['test_label'] = read_idx_labels('/workspaces/KAN-Network/Dataset/t10k-labels.idx1-ubyte') #contains the labels for the testing data
    def __getitem__(self, key):
        return self.ourdataset[key]