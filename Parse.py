import numpy as np

data = []
labels = []

with open("data.csv") as f:
    for line in f:
        parts = line.strip().split(",")
        _id = parts[0]
        label = 1 if parts[1] == "M" else 0
        features = [float(v) for v in parts[2:]]
        data.append(features)
        labels.append(label)


X = np.array(data)
y = np.array(labels)


mean = X.mean(axis=0)
std = X.std(axis=0)
X_norm = (X - mean) / std

a = np.array([[1, 2, 3], [8, 9, 7]])
mean = a.mean(axis=0)
std = a.std(axis=0)
X_norm = (a - mean) / std


split = int(0.8 * len(X_norm))
X_train, X_test = X_norm[:split], X_norm[split:]
y_train, y_test = y[:split], y[split:]

