# Implement GaussianNB in scikit-learn
# create a class Gaussian and the method fit, predict, score just like in Scikit-learn

# Import the libraries
import numpy as np
from datetime import datetime
from scipy.stats import multivariate_normal as mvn


# Define the Gaussian class
class GaussianNB:
    def __init__(self, smoothing=1e-3):
        self.smoothing = smoothing

    def fit(self, X, y):
        K = len(set(y))
        N, D = X.shape

        self.logpriors = np.zeros(K)
        self.means = np.zeros((K, D))
        self.vars = np.zeros((K, D))

        for k in range(K):
            # prior - log(Nk / N) = log(Nk) - log(N)
            self.logpriors[k] = np.log(len(y[y == k])) - np.log(N)

            # Likelihood
            Xk = X[y == k]
            # axis=0 => compute the mean along the column
            self.means[k] = Xk.mean(axis=0)
            # v = variance
            # p(x) = 1/sqrt(2pi.v). e^(-1/2 . (x - mean)^2 / v) => v + epsilon (epsilon = smoothing) to prevent v = 0
            self.vars[k] = Xk.var(axis=0) + self.smoothing

    def score(self, X, y):
        P = self.predict(X)
        return np.mean(P == y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.logpriors)
        P = np.zeros((N, K))

        for k, pr, m, v in zip(range(K), self.logpriors, self.means, self.vars):
            P[:, k] = mvn.logpdf(X, mean=m, cov=v) + pr

        return np.argmax(P, axis=1)


# Apply the class into MNIST dataset from keras
# Import the dataset
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Preprocess the data (convert it to the suitable shape)
# Normalize the data can lead to better performance for GaussianNB <= Reason: The way computer stores floating numbers
X_train = train_X.reshape(-1, 784) / 255.
y_train = train_y

X_test = test_X.reshape(-1, 784) / 255.
y_test = test_y

# Test the model
model = GaussianNB(smoothing=1e-2)

t0 = datetime.now()
model.fit(X_train, y_train)
print("fit duration:", datetime.now() - t0)

t0 = datetime.now()
print("train accuracy:", model.score(X_train, y_train))
print("train predicting duration:", datetime.now() - t0)

t0 = datetime.now()
print("test accuracy:", model.score(X_test, y_test))
print("test predicting duration:", datetime.now() - t0)

'''-----------Exercise-----------'''
# 1) Implement using quadratic form - is it faster?
# 2) Implement LDA - does the quadratic term help or hurt?
