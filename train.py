import json
import math
import numpy as np

T = 1
L1_weights = np.zeros((16, 30))
L2_weights = np.zeros((4, 16))
L3_weights = np.zeros((2, 4))
Lr = 2

L1_biases = np.zeros(16)
L2_biases = np.zeros(4)
L3_biases = np.zeros(2)

l2 = [0] * 16
l3 = [0] * 4
l4 = [0] * 2



def crossEntropy(m, b, lable):
    Cm = m - lable
    Cb = b - 1 - lable
    return Cm, Cb


def softMax(l4):
    total = math.exp(l4[0] / T) + math.exp(l4[1] / T)
    m = math.exp(l4[0] / T) / total
    b = math.exp(l4[1] / T) / total
    return m, b


def activation(z):
    return 1 / (1 + np.exp(-z))


def sigmoidDerevitave(layer, weights, biases):
    result = []

    for i in range(0, len(weights)):
        z = sum(layer * weights[i]) + biases[i]
        neural = activation(z) * (1 - activation(z))
        result.append(neural)
    
    return result



def derivative(delta, Dz, weights):
    gradiant = np.dot(weights.T, delta) * Dz
    return  gradiant


def backPropagation(Cm, Cb, feature):
    global L1_weights, L2_weights, L3_weights, L1_biases, L2_biases, L3_biases

    gradiant = [Cm, Cb]
    gradiant = np.array(gradiant)

    L3_biases -= Lr * gradiant
    gradiant = derivative(gradiant, sigmoidDerevitave(l2, L2_weights, L2_biases),  L3_weights)
    L3_weights -= Lr * (gradiant * l3)
     
    L2_biases -= Lr * gradiant
    gradiant = derivative(gradiant, sigmoidDerevitave(feature, L1_weights, L1_biases), L2_weights)
    L2_weights -= Lr * (gradiant * l2)

    L1_biases -= Lr * gradiant
    gradiant = derivative(gradiant,( activation(feature) * (1 - activation(feature))), L1_weights)
    print(gradiant)
    L1_weights -= Lr * (gradiant * feature)
    # print(L1_weights)


def forwardPropagation(sample):
    global l2, l3, l4

    for i in range(0, len(L1_weights)):
        z = np.dot(sample, L1_weights[i]) + L1_biases[i]
        neural = activation(z)
        l2[i] = neural

    
    for i in range(0, len(L2_weights)):
        z = np.dot(l2, L2_weights[i]) + L2_biases[i]
        neural = activation(z)
        l3[i] = neural


    for i in range(0, len(L3_weights)):
        z = np.dot(l3, L3_weights[i]) + L3_biases[i]
        l4[i] = neural

    M, B = softMax(l4)
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

    for i in  range(0, len(features)):
        M, B = forwardPropagation(features[i])
        Cm, Cb,  = crossEntropy(M, B, lables[i])
        backPropagation(Cm, Cb, features[i])
        # print(L1_weights)

    save_parameters()



