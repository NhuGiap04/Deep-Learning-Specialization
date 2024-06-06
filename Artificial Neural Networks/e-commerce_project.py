# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
def get_data():
    df = pd.read_csv('ecommerce_data.csv')

    # df.head()
    data = df.to_numpy()

    # shuffle data
    np.random.shuffle(data)

    # split features and labels
    X = data[:, :-1]
    Y = data[:, -1].astype(np.int32)

    # Implement One-hot encoding
    N, D = X.shape
    X2 = np.zeros((N, D + 3))
    X2[:, :(D - 1)] = X[:, :(D - 1)] # non-categorical columns

    for n in range(N):
        t = int(X[n, D - 1])
        X2[n, t + D - 1] = 1

    K = len(set(Y))
    Y2 = np.zeros((N, K))
    for n in range(N):
        t = int(Y[n])
        Y2[n, t] = 1

    # Method 2
    # Z = np.zeros(N, 4)
    # Z[np.arange(N), X[:, D - 1].astype(np.int32)] = 1
    # Z[(r1, r2, r3, ...), (c1, c2, c3,...)] = value
    # X2[:, -4:] = Z

    # Assign X2 back to X
    X = X2
    Y = Y2

    # split train and test
    Xtrain = X[:-100]
    Ytrain = Y[:-100]
    Xtest = X[-100:]
    Ytest = Y[-100:]

    # scale the data
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    # Normalize column 1 and 2
    # for i in (1, 2):
    #     m = Xtrain[:, i].mean()
    #     s = Xtrain[:, i].std()
    #     Xtrain[:, i] = (Xtrain[:, i] - m) / s
    #     Xtest[:, i] = (Xtest[:, i] - m) / s

    return Xtrain, Ytrain, Xtest, Ytest


Xtrain, Ytrain, Xtest, Ytest = get_data()


def one_hot_encoding(Y):
    K = len(set(Y))
    N = len(Y)
    Y2 = np.zeros((N, K))
    for n in range(N):
        t = int(Y[n])
        Y2[n, t] = 1
    return Y2


def one_hot_decoding(X):
    return np.argmax(X, axis=1)


def get_binary_data():
    Xtrain, Ytrain, Xtest, Ytest = get_data()
    X2train = Xtrain[Ytrain <= 1]
    Y2train = Ytrain[Ytrain <= 1]
    X2test = Xtest[Ytest <= 1]
    Y2test = Ytest[Ytest <= 1]
    return X2train, Y2train, X2test, Y2test


def get_xor():
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50, 2)) / 2 + 0.5  # (0.5-1, 0.5-1)
    X[50:100] = np.random.random((50, 2)) / 2  # (0-0.5, 0-0.5)
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]])  # (0-0.5, 0.5-1)
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]])  # (0.5-1, 0-0.5)
    Y = np.array([0] * 100 + [1] * 100)
    Y = one_hot_encoding(Y)
    return X, Y


def get_donut():
    N = 200
    R_inner = 5
    R_outer = 10

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = np.random.randn(N // 2) + R_inner
    theta = 2 * np.pi * np.random.random(N // 2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N // 2) + R_outer
    theta = 2 * np.pi * np.random.random(N // 2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer])
    Y = np.array([0] * (N // 2) + [1] * (N // 2))
    Y = one_hot_encoding(Y)
    return X, Y


def get_spiral():
    # Idea: radius -> low...high
    #           (don't start at 0, otherwise points will be "mushed" at origin)
    #       angle = low...high proportional to radius
    #               [0, 2pi/6, 4pi/6, ..., 10pi/6] --> [pi/2, pi/3 + pi/2, ..., ]
    # x = rcos(theta), y = rsin(theta) as usual

    radius = np.linspace(1, 10, 100)
    thetas = np.empty((6, 100))
    for i in range(6):
        start_angle = np.pi * i / 3.0
        end_angle = start_angle + np.pi / 2
        points = np.linspace(start_angle, end_angle, 100)
        thetas[i] = points

    # convert into cartesian coordinates
    x1 = np.empty((6, 100))
    x2 = np.empty((6, 100))
    for i in range(6):
        x1[i] = radius * np.cos(thetas[i])
        x2[i] = radius * np.sin(thetas[i])

    # inputs
    X = np.empty((600, 2))
    X[:, 0] = x1.flatten()
    X[:, 1] = x2.flatten()

    # add noise
    X += np.random.randn(600, 2) * 0.5

    # targets
    Y = np.array([0] * 100 + [1] * 100 + [0] * 100 + [1] * 100 + [0] * 100 + [1] * 100)
    Y = one_hot_encoding(Y)
    return X, Y


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
    X = np.array(X.todense())
    y = np.reshape(y, newshape=(len(y), 1))

    return X, y


# Randomly initialize weights
# M = 5
# D = X.shape[1]
# K = len(set(Y))

# W1 = np.random.randn(D, M)
# b1 = np.zeros(M)
# W2 = np.random.randn(M, K)
# b2 = np.zeros(K)


def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def derivative_sigmoid(X):
    return X * (1 - X)


def tanh(X):
    return np.tanh(X)


def derivative_tanh(X):
    return 1 - X * X


def relu(X):
    return (X >= 0) * X


def derivative_relu(X):
    return X >= 0


def objective_func(X, W_1, b_1, W_2, b_2, T, func=relu, smoothing=1e-3):
    Y = np.log(forward(X, W_1, b_1, W_2, b_2, func) + smoothing)
    N = len(T)
    return 1/N * np.sum(T * Y)


# Implement one-hidden-layer neural network forward propagation
def forward(X, W_1, b_1, W_2, b_2, func=relu):
    Z = func(X.dot(W_1) + b_1)
    return sigmoid(Z.dot(W_2) + b_2)


losses = []


# Implement one-hidden-layer neural network backpropagation
def backpropagation(X, T, num_epochs=1000, lr=1e-5, func=relu, derivative_func=derivative_relu, hidden_size=6, num_class=2):  # T = target, Y = Prediction
    N = len(X)
    K = num_class
    D = X.shape[1]
    M = hidden_size
    # Initialize random weights
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)
    for _ in range(num_epochs):
        Y = forward(X, W1, b1, W2, b2, func)
        # Compute the gradient with respect to W2, b2
        delta2 = T - Y
        Z1 = func(X.dot(W1) + b1)
        grad_W2 = Z1.T.dot(delta2)
        grad_b2 = delta2.T.dot(np.ones(shape=(N,)))
        # Compute the gradient with respect to W1, b1
        delta1 = delta2.dot(W2.T) * derivative_func(Z1)
        grad_W1 = X.T.dot(delta1)
        grad_b1 = delta1.T.dot(np.ones(shape=(N,)))

        # Gradient descent
        W1 += lr * grad_W1
        b1 += lr * grad_b1
        W2 += lr * grad_W2
        b2 += lr * grad_b2

        # Compute the new loss
        loss = objective_func(X, W1, b1, W2, b2, T, func)
        losses.append(loss)

    plt.plot(losses)
    plt.title("loss per iteration")
    plt.show()

    return W1, b1, W2, b2


# Making Predictions
X, Y = spam_emails()


scaler = StandardScaler()
X = scaler.fit_transform(X) / 10

W1, b1, W2, b2 = backpropagation(X, Y, func=relu, derivative_func=derivative_relu, num_class=1)


P_Y_given_X = forward(X, W1, b1, W2, b2)


predictions = np.argmax(P_Y_given_X, axis=1)


# determine the classification rate
# num correct / num total
def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    print(n_correct, n_total)
    return float(n_correct) / n_total


def plot_decision_boundary(model, resolution=100, colors=('b', 'k', 'r')):
    fig, ax = plt.subplots()

    # Generate coordinate grid of shape [resolution x resolution]
    # and evaluate the model over the entire space
    x_range = np.linspace(model.Xtrain[:, 0].min(), model.Xtrain[:, 0].max(), resolution)
    y_range = np.linspace(model.Xtrain[:, 1].min(), model.Xtrain[:, 1].max(), resolution)
    grid = [[model._decision_function(np.array([[xr, yr]])) for yr in y_range] for xr in x_range]
    grid = np.array(grid).reshape(len(x_range), len(y_range))

    # Plot decision contours using grid and
    # make a scatter plot of training data
    ax.contour(x_range, y_range, grid.T, (-1, 0, 1), linewidths=(1, 1, 1),
               linestyles=('--', '-', '--'), colors=colors)
    ax.scatter(model.Xtrain[:, 0], model.Xtrain[:, 1],
               c=model.Ytrain, lw=0, alpha=0.3, cmap='seismic')

    # debug
    ax.scatter([0], [0], c='black', marker='x')

    plt.show()


print("Score:", classification_rate(one_hot_decoding(Y), predictions))