from util import get_xor, get_donut, get_data
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split


def get_data():
    w = np.array([-0.5, 0.5])
    b = 0.1
    X = np.random.random((300, 2)) * 2 - 1
    Y = np.sign(X.dot(w) + b) > 0
    return X, Y

# Perceptron only be used for Binary Classification Problem
class Perceptron:
    def __init__(self):
        self.w = None

    def fit(self, X, Y, learning_rate=0.01, max_epochs=1000):
        N = len(Y)
        bias = np.ones(shape=(len(X), 1))
        X = np.concatenate((bias, X), axis=1)
        Y = np.sign(Y - 0.5)  # Y includes 2 class: {0, 1}
        # Generate a random vector w with length len(X[0]) + 1
        costs = []
        self.w = np.random.rand(len(X[0]))
        for epoch in range(max_epochs):
            incorrect = np.argwhere(np.sign(X.dot(self.w)) != Y)
            if len(incorrect) == 0:
                break
            # Get a random misclassified examples
            incorrect = incorrect.reshape((len(incorrect),))
            index = random.choice(incorrect)
            x, y = X[index], Y[index]
            # Update w
            self.w = self.w + learning_rate * y * x

            c = len(incorrect) / N
            costs.append(c)
        print("Final w:", self.w)
        plt.plot(costs)
        plt.show()

    def predict_one(self, x):
        x = np.concatenate(([1], x), axis=0)
        return np.sign(np.sign(x.dot(self.w)) + 1)

    def score(self, X, Y):
        Y_predict = []
        for test in X:
            Y_predict.append(self.predict_one(test))
        Y_predict = np.array(Y_predict)
        return np.sum(Y_predict == Y) / len(Y)


'''----------------------IMPLEMENTATION----------------------'''
model = Perceptron()

X, Y = get_xor()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

t0 = datetime.now()
model.fit(X_train, Y_train, max_epochs=200)
print("Training time:", (datetime.now() - t0))

t0 = datetime.now()
print("Train accuracy:", model.score(X_train, Y_train))
print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Y_train))


t0 = datetime.now()
print("Test accuracy:", model.score(X_test, Y_test))
print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Y_test))
