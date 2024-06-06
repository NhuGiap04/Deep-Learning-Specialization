# Import the libraries
import numpy as np


# Create the Logistic Class
class LogisticRegression:
    def __init__(self, C=0.1):
        self.C = C  # C is the L2 regularization parameter

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cross_entropy(self, W, b, X, T):
        # Better to use for-loop to avoid invalid value encountered in log when using matrix multiplication
        # W: Weights, b: bias, X: data, T: target
        # W: Dx1, b: 1x1, X: NxD, T:Nx1
        E = 0
        N = len(X)
        for i in range(N):
            if T[i] == 1:
                E -= np.log(self.sigmoid(X[i].dot(W) + b))
            else:
                E -= np.log(1 - self.sigmoid(X[i].dot(W) + b))
        return E + self.C / 2 * W.T.dot(W)

    def predict(self, X, W, b):
        return np.round(self.sigmoid(X.dot(W) + b))

    def fit(self, X, T, learning_rate=1e-3, num_epochs=200):  # Expect X and T are NxD and Nx1 numpy arrays
        N = len(X)  # Number of samples
        D = X.shape[1]  # Number of features

        # Initialize random weights and bias terms
        self.W = np.random.randn(D)
        self.b = np.random.randn(1)

        # Perform Gradient Descent
        for _ in range(num_epochs + 1):
            if _ % 10 == 0:
                print(*self.cross_entropy(self.W, self.b, X, T))
            Y = self.sigmoid(X.dot(self.W) + self.b)  # Get the initial prediction
            self.W += learning_rate * X.T.dot(T - Y) - self.C * self.W
            self.b += learning_rate * np.sum(T - Y, axis=0)

        print("Final W:", self.W)
        print("Final b:", self.b)

    def score(self, X, T):
        prediction = self.predict(X, self.W, self.b)
        return np.mean(prediction == T)


def get_data():
    N = 100
    D = 2

    X = np.random.rand(N, D)

    X[:50, :] = X[:50, :] - 2 * np.ones((50, D))
    X[50:, :] = X[50:, :] + 2 * np.ones((50, D))

    T = np.array([0] * 50 + [1] * 50)

    return X, T


X, T = get_data()

model = LogisticRegression(C=0)
model.fit(X, T, learning_rate=0.1, num_epochs=100)
print(model.score(X, T))



