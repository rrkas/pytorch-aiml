import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

# get data =========================
df = pd.read_csv("data/Salary.csv")
X = df["YearsExperience"].values
Y = df["Salary"].values
D = len(Y)

# preprocessing ====================
x_mean = X.mean()
x_std = X.std()
X = (X - x_mean) / x_std
X = torch.from_numpy(X.reshape(D, 1).astype(np.float32))
print(X.data.shape)

y_mean = Y.mean()
y_std = Y.std()
Y = (Y - y_mean) / y_std
Y = torch.from_numpy(Y.reshape(D, 1).astype(np.float32))
print(Y.data.shape)


# build model ======================
model = nn.Sequential(
    nn.Linear(1, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# train model ======================
losses = []
n_epochs = 100
for i in range(n_epochs):
    optimizer.zero_grad()

    outputs = model(X)
    loss = criterion(outputs, Y)
    losses.append(loss.item())

    loss.backward()
    optimizer.step()

    print(f"Epoch: {i:3d}, loss: {loss.item()}")

plt.plot(losses)

# regression plot =========================
with torch.no_grad():
    predicted = model(X)
    plt.figure()
    plt.plot(X, Y.numpy(), 'ro')
    plt.plot(X, predicted.numpy())

plt.show()
