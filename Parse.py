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
            # Handle dataset with or without labels (1 for M, 0 for B)
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

    # 1. Shuffle dataset reproducibility
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)

    X, y, ids = X[indices], y[indices], ids[indices]

    # 2. Train/Validation Split
    split = int(split_ratio * len(X))
    X_train_raw, X_val_raw = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    ids_val = ids[split:]

    # 3. Fit normalization parameters strictly on X_train
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std == 0] = 1.0  # Prevent division by zero

    # 4. Save mean and std to disk for future predict runs
    np.save("mean.npy", mean)
    np.save("std.npy", std)

    # 5. Transform features using training statistics
    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std

    return X_train, X_val, y_train, y_val, ids_val


def prepare_predict_data(filepath="data.csv"):
    """Loads feature data and applies saved normalization stats."""
    X, y, ids = load_and_preprocess(filepath)

    # Load parameters saved during training
    try:
        mean = np.load("mean.npy")
        std = np.load("std.npy")
    except FileNotFoundError:
        print("Error: Normalization files (mean.npy, std.npy) not found. Run training first.")
        sys.exit(1)

    # Transform data using saved training statistics
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
        train.train(X_train, y_train)

    elif mode in ["predict", "and_predict"]:
        X_test, y_test, ids_test = prepare_predict_data(csv_file)
        predict.predict(X_test, ids_test, y_test)

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)