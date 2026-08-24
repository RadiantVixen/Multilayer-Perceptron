import json
import math
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


def softMax(l4):
    logits = np.exp(np.array(l4) / T)
    total = np.sum(logits)
    m = logits[0] / total
    b = logits[1] / total
    return m, b


def activation(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoidDerevitave(a):
    return activation(a) * (1 - activation(a))

def ft_gradient(delta, a, weights):
    gradient = np.dot(weights.T, delta) * sigmoidDerevitave(a)
    return  gradient


def backPropagation(Cm, Cb, feature):
    global L1_weights, L2_weights, L3_weights, L1_biases, L2_biases, L3_biases

    gradient = np.array([Cm, Cb])

    L3_biases -= Lr * gradient
    L3_weights -= Lr * np.outer(gradient, l3)
    gradient = ft_gradient(gradient, l3,  L3_weights)
    
    L2_biases -= Lr * gradient
    L2_weights -= Lr * np.outer(gradient, l2)
    gradient = ft_gradient(gradient, l2, L2_weights)

    L1_biases -= Lr * gradient
    L1_weights -= Lr * np.outer(gradient, feature)
    # gradient = ft_gradient(gradient, feature, L1_weights)


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

    M, B = softMax(l4)
    print(M, B)
    return M, B



def save_parameters(filepath="parameters.json"):

    params = {
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



def train(features, lables):

    for epoch in range(0, Epochs):
        for i in  range(0, len(features)):
            M, B = forwardPropagation(features[i])
            Cm, Cb,  = crossEntropy(M, B, lables[i])
            backPropagation(Cm, Cb, features[i])

    save_parameters()



