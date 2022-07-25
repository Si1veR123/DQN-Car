"""
Activation functions for Neural Network in neural_network_classes
"""

import numpy as np


def tanh(x, derivative=False):
    if derivative:
        return 1 - (np.tanh(x) ** 2)
    else:
        return np.tanh(x)


def relu(x, derivative=False):
    if derivative:
        return (x > 0).astype(int)
    else:
        return np.maximum(0, x)


def sigmoid(x, derivative=False):
    if derivative:
        val = sigmoid(x)
        return val * (1-val)
    else:
        return 1 / (1 + np.exp(-x))


def softmax(x, outputs):
    return np.exp(x) / np.sum(np.exp(outputs))


def linear(x, derivative=False):
    if derivative:
        return 1
    else:
        return x


def leaky_relu(x, derivative=False):
    if derivative:
        return np.min((x > 0).astype(int) + 0.01, 1)

    return np.maximum(0.01*x, x)
