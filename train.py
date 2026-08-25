import json
import numpy as np

T = 1
L1_weights = np.random.randn(16, 30) * np.sqrt(1.0 / 30)
L2_weights = np.random.randn(4, 16) * np.sqrt(1.0 / 16)
L3_weights = np.random.randn(2, 4) * np.sqrt(1.0 / 4)
Lr = 0.001
Epochs = 100

L1_biases = np.zeros(16)
L2_biases = np.zeros(4)
L3_biases = np.zeros(2)

l2 = [0] * 16
l3 = [0] * 4
l4 = [0] * 2



def crossEntropy(m, b, lable):
    Cm = m - lable
    Cb = b - (1 - lable)
    return Cm, Cb


def softMax(logits):
    logits = np.asarray(logits, dtype=float)
    shifted_logits = logits - np.max(logits)
    exponentials = np.exp(shifted_logits)
    probabilities = exponentials / np.sum(exponentials)

    return probabilities[0], probabilities[1]


def activation(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoidDerivative(a):
    return a * (1 - a)

def ft_gradient(delta, a, weights):
    return np.dot(weights.T, delta) * sigmoidDerivative(a)


def backPropagation(Cm, Cb, feature):
    global L1_weights, L2_weights, L3_weights, L1_biases, L2_biases, L3_biases

    L3_gradient = np.array([Cm, Cb])
    L2_gradient = ft_gradient(L3_gradient, l3,  L3_weights)
    L1_gradient = ft_gradient(L2_gradient, l2, L2_weights)



    L3_biases -= Lr * L3_gradient
    L3_weights -= Lr * np.outer(L3_gradient, l3)
    
    L2_biases -= Lr * L2_gradient
    L2_weights -= Lr * np.outer(L2_gradient, l2)

    L1_biases -= Lr * L1_gradient
    L1_weights -= Lr * np.outer(L1_gradient, feature)


def forwardPropagation(sample):
    global l2, l3, l4

    z = np.dot(L1_weights, sample) + L1_biases
    neural = activation(z)
    l2 = neural

    
    z = np.dot(L2_weights, l2) + L2_biases
    neural = activation(z)
    l3 = neural

    z = np.dot(L3_weights, l3) + L3_biases
    l4 = z

    return softMax(l4)


def cross_entropy_loss(probability_m, label):
    epsilon = 1e-12
    probability_m = np.clip(probability_m, epsilon, 1 - epsilon)
    return -(
        label * np.log(probability_m)
        + (1 - label) * np.log(1 - probability_m)
    )


def evaluate(features, labels):
    total_loss = 0.0
    correct = 0

    for feature, label in zip(features, labels):
        probability_m, _ = forwardPropagation(feature)
        prediction = 1 if probability_m >= 0.5 else 0
        total_loss += cross_entropy_loss(probability_m, label)
        correct += prediction == label

    return total_loss / len(features), correct / len(features)


def plot_learning_curves(history):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Learning curves unavailable. Install dependencies with: python -m pip install -r requirements.txt")
        return

    epochs = range(1, len(history["loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], label="training loss")
    plt.plot(epochs, history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["accuracy"], label="training accuracy")
    plt.plot(epochs, history["val_accuracy"], label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("learning_curves.png")
    plt.show()



def save_parameters(filepath="parameters.json"):

    params = {
        "topology": [30, 16, 4, 2],
        "hidden_activation": "sigmoid",
        "output_activation": "softmax",
        "L1_weights": L1_weights.tolist() if isinstance(L1_weights, np.ndarray) else L1_weights,
        "L2_weights": L2_weights.tolist() if isinstance(L2_weights, np.ndarray) else L2_weights,
        "L3_weights": L3_weights.tolist() if isinstance(L3_weights, np.ndarray) else L3_weights,
        "L1_biases": L1_biases.tolist() if isinstance(L1_biases, np.ndarray) else L1_biases,
        "L2_biases": L2_biases.tolist() if isinstance(L2_biases, np.ndarray) else L2_biases,
        "L3_biases": L3_biases.tolist() if isinstance(L3_biases, np.ndarray) else L3_biases,
    }

    with open(filepath, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Parameters saved to {filepath}")



def train(features, lables, validation_features, validation_labels):
    history = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_accuracy": [],
    }

    for epoch in range(Epochs):
        indices = np.arange(len(features))
        np.random.shuffle(indices)

        for i in indices:
            M, B = forwardPropagation(features[i])
            Cm, Cb = crossEntropy(M, B, lables[i])
            backPropagation(Cm, Cb, features[i])

        loss, accuracy = evaluate(features, lables)
        val_loss, val_accuracy = evaluate(validation_features, validation_labels)
        history["loss"].append(loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"epoch {epoch + 1:03d}/{Epochs} - loss: {loss:.4f} "
            f"- accuracy: {accuracy:.4f} - val_loss: {val_loss:.4f} "
            f"- val_accuracy: {val_accuracy:.4f}"
        )

    save_parameters()
    plot_learning_curves(history)



