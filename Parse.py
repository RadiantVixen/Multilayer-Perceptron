import sys
import numpy as np

# Load your custom model scripts
try:
    import train
    import predict
except ImportError:
    pass

def load_and_preprocess(filepath="data.csv"):
    """Parses raw CSV data into features, labels, and IDs."""
    data = []
    labels = []
    ids = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            
            ids.append(parts[0])
            if parts[1] in ["M", "B"]:
                labels.append(1 if parts[1] == "M" else 0)
            else:
                labels.append(None)
                
            features = [float(v) for v in parts[2:]]
            data.append(features)

    X = np.array(data)
    y = np.array(labels) if labels[0] is not None else None
    ids = np.array(ids)

    return X, y, ids


def prepare_train_data(filepath="data.csv", split_ratio=0.8):
    """Parses, shuffles, splits, normalizes, and saves mean/std stats."""
    X, y, ids = load_and_preprocess(filepath)

    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)

    X, y, ids = X[indices], y[indices], ids[indices]

    split = int(split_ratio * len(X))
    X_train_raw, X_val_raw = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    ids_val = ids[split:]

    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std == 0] = 1.0

    np.save("mean.npy", mean)
    np.save("std.npy", std)

    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std

    return X_train, X_val, y_train, y_val, ids_val


def prepare_predict_data(filepath="data.csv"):
    """Loads feature data and applies saved normalization stats."""
    X, y, ids = load_and_preprocess(filepath)

    try:
        mean = np.load("mean.npy")
        std = np.load("std.npy")
    except FileNotFoundError:
        print("Error: Normalization files (mean.npy, std.npy) not found. Run training first.")
        sys.exit(1)

    X_normalized = (X - mean) / std
    return X_normalized, y, ids


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py [train|predict] [optional_file.csv]")
        sys.exit(1)

    mode = sys.argv[1]
    csv_file = sys.argv[2] if len(sys.argv) > 2 else "data.csv"

    if mode in ["train", "and_train"]:
        X_train, X_val, y_train, y_val, ids_val = prepare_train_data(csv_file)

        print("x_train shape :", X_train.shape)
        print("x_valid shape :", X_val.shape)
        
        train.train(X_train, y_train, X_val, y_val)

    elif mode in ["predict", "and_predict"]:
        X_test, y_test, ids_test = prepare_predict_data(csv_file)
        predict.predict(X_test, ids_test, y_test)

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
