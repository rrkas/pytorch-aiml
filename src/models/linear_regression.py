import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import save_results


class LinearRegression:
    def __init__(self, inp_len, out_len) -> None:
        self.inp_len = inp_len
        self.out_len = out_len

        self.model = nn.Linear(self.inp_len, self.out_len)

        self.criterion = None
        self.optimizer = None

    def prepare_train(self, criterion: nn.MSELoss.__base__, optimizer: Optimizer):
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, X, Y, n_epochs):
        plt.scatter(X, Y)
        save_results.save_plt(plt, "data")

        N = len(X)
        # In ML we want our data to be of shape:
        # (num_samples x num_dimensions)
        X = X.reshape(N, self.inp_len)
        Y = Y.reshape(N, self.out_len)

        # PyTorch uses float32 by default
        # Numpy creates float64 by default
        inputs = torch.from_numpy(X.astype(np.float32))
        targets = torch.from_numpy(Y.astype(np.float32))
        losses = []
        spaces = len(str(n_epochs))
        for i in range(n_epochs):
            # zero the parameter gradients (reset gradient)
            self.optimizer.zero_grad()

            # forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            losses.append(loss.item())

            # backward pass
            loss.backward()  # => g = del(t, L)
            self.optimizer.step()  # => t = t - n * g

            print(
                f"Epoch {str(i + 1).zfill(spaces)}/{n_epochs}, Loss: {loss.item():7.4f}"
            )

        # errors
        plt.figure()
        plt.plot(losses)
        save_results.save_plt(plt, "errors")

        # plot
        plt.figure()
        predicted = self.model(inputs).detach().numpy()
        plt.scatter(X, Y, label="Original data")
        plt.plot(X, predicted, label="Fitted line")
        plt.legend()
        save_results.save_plt(plt, "predicted")

    def parameters(self):
        return self.model.parameters()

    def weight_bias(self):
        return [self.model.weight.data.numpy(), self.model.bias.data.numpy()]
