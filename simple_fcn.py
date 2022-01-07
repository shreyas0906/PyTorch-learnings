# Imports
import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!


# create  FCN

class NN(nn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# model = NN(784, 10)
# x = torch.randn(64, 784)
# print(model(x).shape)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {str(device).upper()}")


# Hyper parameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

#load data
train_dataset = datasets.MNIST(root='MNIST', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='MNIST', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network

for epoch in tqdm(range(num_epochs), desc=f'training'):
    # print(f"Epoch number: {epoch}")
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device=device)

        # reshape the data from ([64, 1, 28, 28]) --> ([64, 784])
        data = data.reshape(data.shape[0], -1)
        # print(f"data shape: {data.shape}    target: {targets.shape}")

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        # print(f"loss: {loss}")


#check accuracy on training and test

def check_accuracy(loader, model):

    if loader.dataset.train:
        print(f"Checking accuracy on training...")
    else:
        print(f"Checking accuracy on testing...")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        acc  = (num_correct/ num_samples) * 100
        print(f"acc: {acc:.3f}")

    model.train()
    return acc


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

