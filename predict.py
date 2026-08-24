import json
import numpy as np
import train


def load_parameters(filepath="parameters.json"):
    with open(filepath, "r") as f:
        params = json.load(f)

    return (
        np.array(params["L1_weights"]),
        np.array(params["L2_weights"]),
        np.array(params["L3_weights"]),
        np.array(params["L1_biases"]),
        np.array(params["L2_biases"]),
        np.array(params["L3_biases"]),
    )


def forwardPropagation(sample, weights_and_biases):
    W1, W2, W3, b1, b2, b3 = weights_and_biases

    z1 = np.dot(W1, sample) + b1
    a1 = train.activation(z1)

    z2 = np.dot(W2, a1) + b2
    a2 = train.activation(z2)

    z3 = np.dot(W3, a2) + b3

    M, B = train.softMax(z3)
    return M, B


def predict(features, ids, y, filepath="parameters.json"):
    params = load_parameters(filepath)

    correct = 0

    for i in range(len(features)):
        M, B = forwardPropagation(features[i], params)

        if M > B:
            prediction = 1
        else:
            prediction = 0

        print(ids[i], end=": ")

        if prediction == 1:
            print("M")
        else:
            print("B")

        if prediction == y[i]:
            correct += 1

    accuracy = correct / len(y) * 100

    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"Correct predictions: {correct}/{len(y)}")