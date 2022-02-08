import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

# get data ========================================
train_dataset = torchvision.datasets.MNIST(
    root="data_downloadable",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
print(f"train set: {train_dataset.data.shape}")
test_dataset = torchvision.datasets.MNIST(
    root="data_downloadable",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)
print(f"test set: {test_dataset.data.shape}")

# build model =====================================
model = nn.Sequential(
    nn.Linear(784, 528),  # layer
    nn.ReLU(),  # process output of layer
    nn.Linear(528, 128),  # layer
    nn.ReLU(),  # process output of layer
    nn.Linear(128, 10),  # hidden layer
)  # No need for final softmax! Its combined with CrossEntropyLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

batch_size = 128

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False,
)

# train model =====================================
n_epochs = 10
c = 0
c_max = n_epochs * len(train_dataset)
c_zfill = len(str(c_max))
n_zfill = len(str(n_epochs))
with open(f"results/ann_mnist.txt", "w") as f:
    for i in range(n_epochs):
        for inputs, targets in train_loader:
            # move data to GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # reshape the input
            inputs = inputs.view(-1, 784)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            c += len(inputs)
            f.write(
                f"Epoch: {str(i + 1).zfill(n_zfill)}/{n_epochs}\t "
                f"c: {str(c).zfill(c_zfill)}/{c_max}\t "
                f"loss: {loss.item()}\n"
            )

# accuracy ========================================
n_correct = 0
n_total = 0
for inputs, targets in test_loader:
    # move data to GPU
    inputs, targets = inputs.to(device), targets.to(device)

    # reshape the input
    inputs = inputs.view(-1, 784)

    outputs = model(inputs)
    _, predictions = torch.max(outputs, 1)

    # update counts
    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]

accuracy = n_correct / n_total
print(f"Accuracy: {accuracy:.4f}")
