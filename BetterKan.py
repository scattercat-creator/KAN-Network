from kan import KAN

# Train on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((14, 14), antialias=True),  # Antialiasing to preserve information
    transforms.Normalize((0.5,), (0.5,))
])

#Full Datasets
full_trainset = torchvision.datasets.MNIST(
    root="./Dataset", train=True, download=True, transform=transform
)
full_valset = torchvision.datasets.MNIST(
    root="./Dataset", train=False, download=True, transform=transform
)

# Subset Datasets
num_train = len(full_trainset)
indices = np.random.permutation(num_train)[:int(num_train / 10)]
trainset = Subset(full_trainset, indices)

num_val = len(full_valset)
val_indices = np.random.permutation(num_val)[:int(num_val / 10)]
valset = Subset(full_valset, val_indices)

print(f"Using {len(trainset)} training samples out of {num_train} total")
print(f"Using {len(valset)} validation samples out of {num_val} total")

#Functions
def evaluate(testloader, model):
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.view(images.size(0), -1).to(device)
            output = model(images)
            val_loss += criterion(output, labels.to(device)).item()
            val_accuracy += (
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
    val_loss /= len(testloader)
    val_accuracy /= len(testloader)
    return val_loss, val_accuracy

#Variables
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

input_size = 14*14  
model = KAN([input_size, 64, 10], ckpt_path="./efficient_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
criterion = nn.CrossEntropyLoss()
#Training
for epoch in range(10):
    model.train()
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images = images.view(images.size(0), -1).to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

    model.eval()
    val_loss, val_accuracy = evaluate(valloader, model)
    scheduler.step()
    print(f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

#Save model
model.saveckpt(path='./efficient_model_checkpoint')
