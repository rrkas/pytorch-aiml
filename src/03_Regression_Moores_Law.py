import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import math

from downloader import download_from_url, download_path

# load data ===========================================
url = "https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv"
download_from_url(url, "moore.csv")
df = pd.read_csv(f"{download_path}/moore.csv", header=None).values
X = df[:, 0].reshape(-1, 1)
Y = df[:, 1].reshape(-1, 1)

# exploratory analysis =================================
# plt.scatter(X, Y)

Y = np.log(Y)
# plt.figure()
# plt.scatter(X, Y)

# normalize =============================================
x_mean = X.mean()
x_std = X.std()
X = (X - x_mean) / x_std

y_mean = Y.mean()
y_std = Y.std()
Y = (Y - y_mean) / y_std

# plt.figure()
# plt.scatter(X, Y)

# build model ============================================
inputs = torch.from_numpy(X.astype(np.float32))
targets = torch.from_numpy(Y.astype(np.float32))
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.7)

# train model ===========================================
n_epochs = 100
zfill = len(str(n_epochs))
losses = []
with open(f"results/moore.txt", 'w') as f:
    for i in range(n_epochs):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        f.write(f"Epoch: {str(i + 1).zfill(zfill)}/{n_epochs}\t Loss: {loss.item():.4f}\n")

# plt.figure()
# plt.plot(losses)

with torch.no_grad():
    plt.figure()
    predicted_results = model(inputs)
    plt.plot(X, Y, 'ro', label='Original data')
    plt.plot(X, predicted_results, label='Fitted line')
    plt.legend()

"""
Original:
    C = C0 * (r ** t)

After Log:
    log(C) = log(C0) + t * log(r)
    y = mx + c
        y = log(C)
        m = log(r)
        x = t
        c = log(C0)

After normalization:
    y' = (y - y_mean) / y_std
    x' = (x - x_mean) / x_std
    
    y' = wx' + b
    y = (w * y_std / x_std)x + ( -(w * y_std * x_mean / x_std) + (b * y_std) + y_mean)

    m = w * y_std / x_std
    r = exp(m)
    
time taken for C = 2 * C0:
    2 * C0 = C0 * (r ** t)
    r ** t = 2
    t = log(2) / log(r) = log(2) / m
"""

w = model.weight.data.numpy()[0, 0]
m = w * y_std / x_std
r = math.e ** m
print(f"r = {r}")

t = math.log(2) / m
print(f"t = {t}")

plt.show()
