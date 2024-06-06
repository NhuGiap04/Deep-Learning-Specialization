import numpy as np

N = 100
D = 2

X = np.random.randn(N, D)
X_bias = np.concatenate((np.ones(shape=(N, 1)), X), axis=1)

w = np.random.randn(D + 1)


def sigmoid(X_input, weight):
    z = np.dot(X_input, weight)
    return 1 / (1 + np.exp(-z))


print(sigmoid(X_bias, w))