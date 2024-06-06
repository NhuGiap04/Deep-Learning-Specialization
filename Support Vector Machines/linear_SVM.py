# Import the Library
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from util import get_spiral, get_xor, get_donut, get_clouds

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Define class SVM
class LinearSVM:
    def __init__(self, C=1.0):
        self.C = C

    def _objective(self, margins):
        return 0.5 * self.w.dot(self.w) + self.C * np.maximum(0, 1 - margins).sum()

    def fit(self, X, y, lr=1e-5, n_iters=400):
        N, D = X.shape
        self.N = N
        self.w = np.random.rand(D)
        self.b = 0

        # Gradient Descent
        losses = []
        for _ in range(n_iters):
            margins = y * self._decision_function(X)
            loss = self._objective(margins)
            losses.append(loss)

            idx = np.where(margins < 1)[0]
            grad_w = self.w - self.C * y[idx].dot(X[idx])
            self.w -= lr * grad_w
            grad_b = -self.C * y[idx].sum()
            self.b -= lr * grad_b

        self.support_ = np.where((y * self._decision_function(X)) == 1)[0]
        print("num SVs:", len(self.support_))

        print("w:", self.w)
        print("b:", self.b)

        # hist of margins
        # m = y * self._decision_function(X)
        # plt.hist(m, bins=20)
        # plt.show()

        plt.plot(losses)
        plt.title('losses per iteration')
        plt.show()

    def _decision_function(self, X):
        return X.dot(self.w) + self.b

    def predict(self, X):
        return np.sign(self._decision_function(X))

    def score(self, X, y):
        P = self.predict(X)
        return np.mean(y == P)


def plot_decision_boundary(model, X, Y, resolution=100, colors=('b', 'k', 'r')):
    # np.warnings.filterwarnings('ignore')
    fig, ax = plt.subplots()

    # Generate coordinate grid of shape [resolution x resolution]
    # and evaluate the model over the entire space
    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), resolution)
    y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), resolution)
    grid = [[model._decision_function(np.array([[xr, yr]])) for yr in y_range] for xr in x_range]
    grid = np.array(grid).reshape(len(x_range), len(y_range))

    # Plot decision contours using grid and
    # make a scatter plot of training data
    ax.contour(x_range, y_range, grid.T, (-1, 0, 1), linewidths=(1, 1, 1),
               linestyles=('--', '-', '--'), colors=colors)
    ax.scatter(X[:, 0], X[:, 1],
               c=Y, lw=0, alpha=0.3, cmap='seismic')

    # Plot support vectors (non-zero alphas)
    # as circled points (line-width > 0)
    mask = model.support_
    ax.scatter(X[:, 0][mask], X[:, 1][mask],
               c=Y[mask], cmap='seismic')

    # debug
    ax.scatter([0], [0], c='black', marker='x')

    # debug
    # x_axis = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    # w = model.w
    # b = model.b
    # # w[0]*x + w[1]*y + b = 0
    # y_axis = -(w[0]*x_axis + b)/w[1]
    # plt.plot(x_axis, y_axis, color='purple')
    # margin_p = (1 - w[0]*x_axis - b)/w[1]
    # plt.plot(x_axis, margin_p, color='orange')
    # margin_n = -(1 + w[0]*x_axis + b)/w[1]
    # plt.plot(x_axis, margin_n, color='orange')

    plt.show()


def medical():
    data = load_breast_cancer()
    X, Y = data.data, data.target
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    return Xtrain, Xtest, Ytrain, Ytest, 1e-3, 200

def xor():
    X, Y = get_xor()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    return Xtrain, Xtest, Ytrain, Ytest, 1e-2, 300


def donut():
    X, Y = get_donut()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    return Xtrain, Xtest, Ytrain, Ytest, 1e-2, 300


def spiral():
    X, Y = get_spiral()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    return Xtrain, Xtest, Ytrain, Ytest, 1e-2, 300


def clouds():
    X, Y = get_clouds()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    return Xtrain, Xtest, Ytrain, Ytest, 1e-5, 400


def spam_emails():
    # File "Spam.csv" contains some invalid characters
    # An error may be thrown, so we need to encode it
    df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

    # After encoding, there are some redundant columns
    # Drop unnecessary columns
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

    # rename columns
    df.columns = ['labels', 'data']

    # create binary labels
    df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
    y = df['b_labels'].values

    # try multiple ways of calculating features
    # decode_error: ignore any invalid UTF character
    tfidf = TfidfVectorizer(decode_error='ignore')
    X = tfidf.fit_transform(df['data'])

    # count_vectorizer = CountVectorizer(decode_error='ignore')
    # X = count_vectorizer.fit_transform(df['data'])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    X_train = np.asarray(X_train.todense())
    X_test = np.asarray(X_test.todense())

    return X_train, X_test, y_train, y_test, 1e-3, 200


if __name__ == '__main__':
    Xtrain, Xtest, Ytrain, Ytest, lr, n_iters = spam_emails()
    print("Possible labels:", set(Ytrain))

    # make sure the targets are (-1, +1)
    Ytrain[Ytrain == 0] = -1
    Ytest[Ytest == 0] = -1

    # scale the data
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    # now we'll use our custom implementation
    model = LinearSVM(C=1.0)

    t0 = datetime.now()
    model.fit(Xtrain, Ytrain, lr=lr, n_iters=n_iters)
    print("train duration:", datetime.now() - t0)
    t0 = datetime.now()
    print("train score:", model.score(Xtrain, Ytrain), "duration:", datetime.now() - t0)
    t0 = datetime.now()
    print("test score:", model.score(Xtest, Ytest), "duration:", datetime.now() - t0)

    if Xtrain.shape[1] == 2:    # Can only display if the dataset has only 2 features
        plot_decision_boundary(model, Xtrain, Ytrain)
