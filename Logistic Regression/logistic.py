import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.rand(N, D)

X[:50, :] = X[:50, :] - 2 * np.ones((50, D))
X[50:, :] = X[50:, :] + 2 * np.ones((50, D))

T = np.array([0] * 50 + [1] * 50)

ones = np.array([[1] * N]).T
Xb = np.concatenate((ones, X), axis=1)

# randomly initialize the weights
w = np.random.randn(D + 1)

# Calculate the model output
z = Xb.dot(w)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


y = sigmoid(z)


def cross_entropy(T, y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(y[i])
        else:
            E -= np.log(1 - y[i])
    return E


# Gradient descent
learning_rate = 0.1
for i in range(200):
    if i % 10 == 0:
        print(Xb.dot(w))
        print(cross_entropy(T, y))

    w += learning_rate * np.dot(Xb.T, T - y)
    y = sigmoid(Xb.dot(w))

print("Final w:", w)

# plt.scatter(X[:, 0], X[:, 1], c=T, s=100, alpha=0.5)
#
# x_axis = np.linspace(-6, 6, 100)
# y_axis = -x_axis
# plt.plot(x_axis, y_axis)
# plt.show()

'''-------------Homework--------------:
1. Implement the L2 Regularization
2. Implement the L1 Regularization
'''