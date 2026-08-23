import math
import numpy as np

T = 1
L1_wieghts = [[0] * 30] * 16
L2_wieghts = [[0] * 16] * 4
L3_wieghts = [[0] * 4] * 2

L1_bias = [0] * 16
L2_bias = [0] * 4
L3_bias = [0] * 2

l2 = [] * 16
l3 = [] * 4
l4 = [] * 1



def crossEntropy(m, b, lable):
    Cm = m - lable
    Cm = b - 1 - lable


def softMax(l4):
    total = math.exp(l4[0] / T) + math.exp(l4[1] / T)
    m = math.exp(l4[0] / T) / total
    b = math.exp(l4[1] / T) / total
    return m, b


def activation(z):
    return (1 / (1 + math.exp(z * -1)))


def sigmoidDerevitave(layer, wieghts, biases):
    result = []

    for i in range(0, len(wieghts)):
        z = layer * wieghts[i]
        z + biases[i]
        neural = activation(z) * (1 - activation(z))
        result.append(neural)
    return result



def delta(oldDelta, layer, wieghts, biases):
    return (oldDelta ** T) * oldDelta ** sigmoidDerevitave(layer, wieghts, biases)



def derivative(delta, layer, wieghts, biases, a):
    gradiant = delta(delta, layer, wieghts, biases)
    return  gradiant * a,  gradiant


def backPropagation(Cm, Cb, feature):
    gradiant = [Cm, Cb]

    for i in range(0, L1_wieghts):
        w, b = derivative(gradiant, l4, L1_wieghts, L1_biases, l3)
        L1_wieghts -= Lr * w
        L1_biases -= Lr * b
    
    for i in range(0, L2_wieghts):
        w, b = derivative(gradiant, l3, L2_wieghts, L2_biases, l2)
        L2_wieghts -= Lr * w
        L2_biases -= Lr * b

    for i in range(0, L3_wieghts):
        w, b = derivative(gradiant, l2, L3_wieghts, L3_biases, feature)
        L3_wieghts -= Lr * w
        L3_biases -= Lr * b

def forwardPropagation(sample):

    for i in range(0, len(L1_wieghts)):
        z = sample * L1_wieghts[i]
        z + L1_bias[i]
        neural = activation(z)
        l2.append(neural)

    
        neural = activation(z)
    for i in range(0, len(L2_wieghts)):
        z = sample * L2_wieghts[i]
        z + L2_bias[i]
        neural = activation(z)
        l3.append(neural)


    for i in range(0, len(L3_wieghts)):
        z = sample * L3_wieghts[i]
        z + L3_bias[i]
        l4.append(neural)

    M, B = softMax(l4)
    return M, B


def train(features, lables):

    for i in  range(0, len(features)):
        M, B = forwardPropagation(features[i])
        Cm, Cb,  = crossEntropy(M, B, lables[i])
        backPropagation(cm, cb, features[i])




