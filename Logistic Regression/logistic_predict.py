from logistic_data_preprocessing import get_binary_data
import numpy as np
# import pandas as pd

X, y, _, _ = get_binary_data()

# randomly initialize weight
D = X.shape[1]
w = np.random.randn(D)
b = 0


# make predictions
def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def forward(X, w, b):
    return sigmoid(X.dot(w) + b)


P_y_given_X = forward(X, w, b)
predictions = np.round(P_y_given_X)
print(predictions)


def classification_rate(y, P):
    return np.mean(y == P)


print(classification_rate(y, predictions))
