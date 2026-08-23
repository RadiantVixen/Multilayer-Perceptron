import numpy as np
import train
import json


L1_weights = np.zeros((16, 30))
L2_weights = np.zeros((4, 16))
L3_weights = np.zeros((2, 4))

L1_biases = [0] * 16
L2_biases = [0] * 4
L3_biases = [0] * 2


def load_parameters(filepath="parameters.json"):
    global L1_weights, L2_weights, L3_weights, L1_biases, L2_biases, L3_biases


    with open(filepath, "r") as f:
        params = json.load(f)
    
    L1_weights = np.array(params["L1_weights"])
    L2_weights = np.array(params["L2_weights"])
    L3_weights = np.array(params["L3_weights"])
    L1_biases = np.array(params["L1_biases"])
    L2_biases = np.array(params["L2_biases"])
    L3_biases = np.array(params["L3_biases"])


def forwardPropagation(sample):
    l2 = []
    l3 = []
    l4 = []
    
    for i in range(0, len(L1_weights)):
        z = np.dot(sample, L1_weights[i]) + L1_biases[i]
        neural = train.activation(z)
        l2.append(neural)

    
    for i in range(0, len(L2_weights)):
        z = np.dot(l2, L2_weights[i]) + L2_biases[i]
        neural = train.activation(z)
        l3.append(neural)


    for i in range(0, len(L3_weights)):
        z = np.dot(l3, L3_weights[i]) + L3_biases[i]
        l4.append(z)


    M, B = train.softMax(l4)
    return M, B




def predict(features, ids):
    global L1_weights, L2_weights, L3_weights, L1_biases, L2_biases, L3_biases

    load_parameters()

    for i in  range(0, len(features)):
        M, B = forwardPropagation(features[i])
        print(ids[i], f"M: {M}", f"B: {B}")
        # print(ids[i], end=": ")
        # if M > B:
        #     print("M")
        # else:
        #     print("B")

