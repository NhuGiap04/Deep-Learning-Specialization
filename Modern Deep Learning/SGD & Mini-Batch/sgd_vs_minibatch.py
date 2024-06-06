import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import time


def get_data():
    print("Reading and Transforming MNIST data")
    df = pd.read_csv('MNIST.csv')
    X = df.iloc[:, 1:].values
    Y = df.iloc[:, 0].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    return X_train, X_test, Y_train, Y_test


def one_hot_encoder(T):
    N = T.shape[0]
    K = max(T) + 1
    Y = np.zeros((N, K))
    for index in range(N):
        Y[index][T[index]] = 1
    return Y


class MultiLogistic():
    def __init__(self):
        return

    def fit_GD(self, X, T, lr=1e-3, num_epochs=200, plot=False):  # X: NxD, Y:NxK where K is the number of targets
        N, D = X.shape
        K = T.shape[1]
        self.W = np.random.randn(D, K) / np.sqrt(D)
        self.b = np.zeros((K,))
        start_time = time.time()
        costs = []
        for _ in range(num_epochs):
            Y = self.predict_prob(X)
            self.W -= lr * X.T.dot(Y - T)
            self.b -= lr * np.sum(Y - T, axis=0)
            costs.append(self.cost(T, Y))
        if plot:
            plt.title('Full Gradient Descent')
            plt.plot(costs)
            plt.show()

        end_time = time.time()
        print("Elapsed time for Full Gradient Descent:", end_time - start_time)

    def fit_mini_batch(self, X, T, lr=1e-3, num_epochs=100, batch_size=128, plot=False):
        N, D = X.shape
        K = T.shape[1]
        self.W = np.random.randn(D, K) / np.sqrt(D)
        self.b = np.zeros((K,))
        costs = []
        start_time = time.time()
        # Implement Mini-Batch Gradient Descent
        for i in range(num_epochs):
            data = np.concatenate((X, T), axis=1)
            np.random.shuffle(data)
            X = data[:, :-K]
            T = data[:, -K:]
            num_batches = int(np.ceil(N / batch_size))
            for j in range(num_batches):
                Xb = X[j * batch_size: (j + 1) * batch_size, :]
                Tb = T[j * batch_size: (j + 1) * batch_size]
                # Calculate gradient and do gradient descent
                Yb = self.predict_prob(Xb)
                self.W -= lr * Xb.T.dot(Yb - Tb)
                self.b -= lr * np.sum(Yb - Tb, axis=0)
            Y = self.predict_prob(X)
            cost = self.cost(T, Y)
            costs.append(cost)
        end_time = time.time()
        print("Elapsed time for Mini-Batch:", end_time - start_time)
        if plot:
            plt.title('Mini-Batch GD')
            plt.plot(costs)
            plt.show()

    def fit_SGD(self, X, T,  lr=1e-3, num_epochs=100, plot=False):
        N, D = X.shape
        K = T.shape[1]
        self.W = np.random.randn(D, K) / np.sqrt(D)
        self.b = np.zeros((K,))
        costs = []
        start_time = time.time()
        # Implement Mini-Batch Gradient Descent
        for i in range(num_epochs):
            data = np.concatenate((X, T), axis=1)
            np.random.shuffle(data)
            X = data[:, :-K]
            T = data[:, -K:]
            for j in range(N):
                X_j = X[j].reshape((1, D))
                Y_j = self.predict_prob(X_j)
                self.W -= lr * X_j.T.dot(Y_j - T[j])
                self.b -= lr * np.sum(Y_j - T[j], axis=0)
            Y = self.predict_prob(X)
            costs.append(self.cost(T, Y))
        end_time = time.time()
        print("Elapsed time for SGD:", end_time - start_time)
        if plot:
            plt.title('SGD')
            plt.plot(costs)
            plt.show()

    def cost(self, T, Y):
        return -np.sum(T * np.log(Y))

    def predict_prob(self, X):
        a = np.exp(X.dot(self.W) + self.b)
        return a / np.sum(a, axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_prob(X), axis=1)

    def score(self, X, T):
        prediction = self.predict(X)
        return np.mean(prediction == T)


def main():
    X_train, X_test, Y_train, Y_test = get_data()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    Y_train_encoded = one_hot_encoder(Y_train)

    model = MultiLogistic()

    model.fit_mini_batch(X_train, Y_train_encoded, lr=1e-4, num_epochs=50, batch_size=2048)
    print("Training score for Mini-Batch:", model.score(X_train, Y_train))
    print("Test score for Mini-Batch:", model.score(X_test, Y_test))

    model.fit_GD(X_train, Y_train_encoded, lr=3 * 1e-5, num_epochs=50)
    print("Training score for Full GD:", model.score(X_train, Y_train))
    print("Test score for Full GD:", model.score(X_test, Y_test))

    model.fit_SGD(X_train, Y_train_encoded, lr=3 * 1e-5, num_epochs=50)
    print("Training score for Full GD:", model.score(X_train, Y_train))
    print("Test score for Full GD:", model.score(X_test, Y_test))


if __name__ == "__main__":
    main()
