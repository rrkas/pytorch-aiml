import numpy as np
import torch
import torch.nn as nn

from models.linear_regression import LinearRegression

# we would like to generate 20 data points
N = 20

# random data on the x-axis in (-5, +5)
X = np.random.random(N) * 10 - 5

# a line plus some noise
w, b = 0.5, -1
Y = w * X + b + np.random.randn(N)

model = LinearRegression(1, 1)
model.prepare_train(nn.MSELoss(), torch.optim.SGD(model.parameters(), lr=0.1))
model.train(X, Y, 30)
print(model.weight_bias())
