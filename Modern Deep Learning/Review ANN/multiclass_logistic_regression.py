import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

    def fit(self, X, T, lr=1e-3, num_epochs=200):  # X: NxD, Y:NxK where K is the number of targets
        N, D = X.shape
        K = T.shape[1]
        self.W = np.random.randn(D, K) / np.sqrt(D)
        self.b = np.zeros((K, ))
        costs = []
        for _ in range(num_epochs):
            Y = self.predict_prob(X)
            self.W -= lr * X.T.dot(Y - T)
            self.b -= lr * np.sum(Y - T, axis=0)
            if _ % 10 == 0:
                costs.append(self.cost(T, Y))
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
    model.fit(X_train, Y_train_encoded, lr=3*1e-5)

    print("Training score:", model.score(X_train, Y_train))
    print("Test score:", model.score(X_test, Y_test))


if __name__ == "__main__":
    main()
