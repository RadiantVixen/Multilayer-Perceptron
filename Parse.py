import sys
import numpy as np
import train
import predict

def parse():
    data = []
    labels = []
    ids = []

    with open("data.csv") as f:
        for line in f:
            parts = line.strip().split(",")
            ids.append(parts[0])
            label = 1 if parts[1] == "M" else 0
            features = [float(v) for v in parts[2:]]
            data.append(features)
            labels.append(label)


    X = np.array(data)
    y = np.array(labels)


    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_norm = (X - mean) / std



    split = int(0.8 * len(X_norm))
    X_train, X_test = X_norm[:split], X_norm[split:]
    y_train, y_test = y[:split], y[split:]
    ids = ids[split:]
    return X_train, X_test, y_train, y_test, ids





if __name__ == "__main__":
    X_train, X_test, y_train, y_test, ids = parse()

    # Check if a command line argument was supplied
    if len(sys.argv) < 2:
        print("Usage: python script.py [train|predict]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "and_train":
        train.train(X_train, y_train)
    elif mode == "and_predict":
        # Call your prediction logic here instead of training
        predict.predict(X_test, ids)
    else:
        print(f"Unknown mode: {mode}")


