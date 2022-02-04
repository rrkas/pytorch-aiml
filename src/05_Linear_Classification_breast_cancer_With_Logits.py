import torch
import numpy as np
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# get data ======================================
data = load_breast_cancer()

# preprocessing ==================================
X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, test_size=0.3)
N, D = X_train.shape

scaler = StandardScaler()

X_train = torch.from_numpy(scaler.fit_transform(X_train).astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32).reshape(-1, 1))

X_test = torch.from_numpy(scaler.transform(X_test).astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32).reshape(-1, 1))

# build model ====================================
model = nn.Sequential(
    nn.Linear(D, 1),
    nn.Sigmoid(),
)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

# train ==========================================
n_epochs = 1000
zfill = len(str(n_epochs))
with open("results/breast_cancer_bce.txt", "w") as f:
    for i in range(n_epochs):
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, Y_train)

        loss.backward()
        optimizer.step()

        f.write(f"Epoch: {str(i + 1).zfill(zfill)}/{n_epochs}\t Loss: {loss.item()} \n")

# accuracy =======================================
with torch.no_grad():
    test_outputs = np.round(model(X_test).numpy())
    accuracy = np.mean(test_outputs == Y_test.numpy())
    print(f"Accuracy = {accuracy}")
