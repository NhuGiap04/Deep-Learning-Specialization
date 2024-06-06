'''---------Implement Artificial Neural Network to work with XOR and Donut problems----------'''
# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class Layer:
    def __init__(self, W, b, activation, derivative_activation=None):
        self.W = W
        self.b = b
        self.activation = activation
        self.derivative_activation = derivative_activation


# Define the necessary Activation Function
# 1. ReLU(Rectified Linear Unit)
def relu(Z):
    relu_Z = tf.math.greater_equal(Z, tf.constant([0.], dtype=tf.float64))
    # Convert relu_Z to type float64 since tf.multiply doesn't accept type Boolean
    relu_Z = tf.cast(relu_Z, dtype=tf.float64)
    return tf.math.multiply(relu_Z, Z)


# 2. Sigmoid
def sigmoid(Z):
    return tf.cast(tf.nn.sigmoid(Z), dtype=tf.float64)


# 3. Tanh
def tanh(Z):
    return tf.cast(tf.nn.tanh(Z), dtype=tf.float64)


# 4. Softmax
def softmax(Z):
    return tf.cast(tf.nn.softmax(Z), dtype=tf.float64)


# 5. Compute derivatives of Activation functions manually
def derivative_sigmoid(Z):
    return tf.multiply(Z, 1 - Z)


def derivative_tanh(Z):
    return 1 - tf.multiply(Z, Z)


def derivative_relu(Z):
    relu_Z = tf.math.greater_equal(Z, tf.constant([0.], dtype=tf.float64))
    return tf.cast(relu_Z, dtype=tf.float64)


# 6. Compute derivatives of a general Activation Function
def derivative(Z, activation):
    with tf.GradientTape() as g:
        g.watch(Z)
        Y = activation(Z)
    dy_dz = g.gradient(Y, Z)
    return tf.cast(dy_dz, dtype=tf.float64)


class ANN:
    def __init__(self, X, T):
        # Get the train data
        self.X = X
        self.T = T  # Expect T haven't been one-hot-encoded
        self.Y = None  # Y => predict output
        self.N = len(X)  # The number of samples
        self.D = X.shape[1]  # The number of features of X
        self.K = len(set(T))  # The number of classes to predict
        # Define the number of layers, weight lists and layer nodes
        self.num_layers = 0
        self.layer_list = []
        self.hidden_nodes = [tf.convert_to_tensor(self.X,
                                                  dtype=tf.float64)]  # Keep the hidden nodes list for future use (backpropagation with recursion delta)

    # Define the one-hot-encoding and one-hot-decoding function
    def one_hot_encoding(self, Z):  # Expect Z as numpy 1D-array
        K = len(set(Z))
        N = len(Z)
        Z2 = np.zeros((N, K))
        for n in range(N):
            t = int(Z[n])
            Z2[n, t] = 1
        return Z2

    def one_hot_decoding(self, Z):  # Expect Z as a one-hot-encoding numpy array
        return np.argmax(Z, axis=1)

    # Define the function to allow to add layers for the Network
    def add_layer(self, hidden_size=5, activation=relu, derivative_activation=None):
        if self.num_layers == 0:
            # Define random weights and biases for input layer->first hidden layer
            W = tf.random.normal(shape=[self.D, hidden_size], dtype=tf.float64)
            W = W / tf.reduce_max(tf.math.abs(W))
            b = tf.random.normal(shape=[1, hidden_size], dtype=tf.float64)
            b = b / tf.reduce_max(tf.math.abs(b))
            # Create new layer
            new_layer = Layer(W, b, activation, derivative_activation)
            self.hidden_nodes.append(
                activation(tf.add(tf.matmul(tf.convert_to_tensor(self.X, dtype=tf.float64), W), b)))
            self.layer_list.append(new_layer)
            self.num_layers += 1
        else:
            previous_hidden_size = self.hidden_nodes[-1].shape[1]
            # Define random weights and biases for last current hidden layer-> new hidden layer or output layer
            W = tf.random.normal(shape=[previous_hidden_size, hidden_size], dtype=tf.float64)
            W = W / tf.reduce_max(tf.math.abs(W))
            b = tf.random.normal(shape=[1, hidden_size], dtype=tf.float64)
            b = b / tf.reduce_max(tf.math.abs(b))
            # Create new layer
            new_layer = Layer(W, b, activation, derivative_activation)
            self.hidden_nodes.append(activation(tf.add(tf.matmul(self.hidden_nodes[-1], W), b)))
            print(self.hidden_nodes[-1].dtype)
            self.layer_list.append(new_layer)
            self.num_layers += 1

    # Define the forward function
    def forward(self, X, updated=False):
        Z = tf.convert_to_tensor(X, dtype=tf.float64)
        for i in range(self.num_layers):
            layer = self.layer_list[i]
            Z = layer.activation(tf.add(tf.matmul(Z, layer.W), layer.b))
            if updated:
                self.hidden_nodes[i + 1] = Z
        return Z

    # Define the cost function:
    def cost(self, X, T, C=0.01):
        Y = self.forward(X)  # Get the predicted output
        regularization_term = 0
        for layer in self.layer_list:
            regularization_term += tf.reduce_sum(layer.W * layer.W).numpy()
        return tf.reduce_sum(1 / self.N * tf.math.multiply(T, tf.math.log(
            Y))).numpy() - (C / 2 * 1 / self.N) * regularization_term  # Cost = T.log(Y) - C/2 * W^2

    # Define the predict function
    def classification_rate(self, Y,
                            T):  # Expect Y is a one-hot-encoding tensor, T is the correct output as numpy 1D-array
        Y = Y.numpy()  # convert Y to numpy array
        Y = self.one_hot_decoding(Y)  # Decode Y
        n_correct = 0
        n_total = 0
        for i in range(len(Y)):
            n_total += 1
            if Y[i] == T[i]:
                n_correct += 1
        print(n_correct, n_total)
        return float(n_correct) / n_total

    def get_info(self):
        print("-----Training dataset-----")
        print(self.X)
        print("-----Test dataset-----")
        print(self.T)
        for i in range(len(self.layer_list)):
            print(f"Weight {i + 1}:\n{self.layer_list[i].W}")
            print(f"Bias {i + 1}:\n{self.layer_list[i].b}")
            print(f"Activation Function corresponding to weight {i + 1}: {self.layer_list[i].activation}")
        for i in range(len(self.hidden_nodes)):
            print(f"Layer {i}:\n{self.hidden_nodes[i]}")

    # Define the backpropagation function
    # Problem: The Gradient Tape compute gradient incorrectly compared to manually compute gradient by matrix multiplication
    def backpropagation(self, learning_rate=1e-6, n_iters=400, C=0.1):
        tensor_T = tf.convert_to_tensor(self.one_hot_encoding(self.T),
                                        dtype=tf.float64)  # Convert the self.T to one-hot-encoding tensor

        losses = []
        for _ in range(n_iters):
            self.Y = self.forward(self.X, updated=True)  # Feedforward to get the prediction output as Y
            # Define the gradient of Weights and gradient of biases
            grad_W = []
            grad_B = []

            # Implement Backpropagation => recursive delta
            delta = tf.subtract(tensor_T, self.Y)
            grad_w = tf.matmul(tf.transpose(self.hidden_nodes[-2]), delta) - C * self.layer_list[-1].W
            grad_b = tf.matmul(tf.ones(shape=(1, self.N), dtype=tf.float64), delta)

            grad_W.append(grad_w)
            grad_B.append(grad_b)

            for i in range(self.num_layers - 1, 0, -1):
                delta = tf.multiply(tf.matmul(delta, tf.transpose(self.layer_list[i].W)),
                                    self.layer_list[i-1].derivative_activation(self.hidden_nodes[i]))
                grad_w = tf.matmul(tf.transpose(self.hidden_nodes[i - 1]), delta) - C * self.layer_list[i - 1].W
                grad_b = tf.matmul(tf.ones(shape=(1, self.N), dtype=tf.float64), delta)

                grad_W.append(grad_w)
                grad_B.append(grad_b)

            # Gradient ascent
            index = 0
            while len(grad_W) > 0 and len(grad_b) > 0:
                grad_w = grad_W.pop()
                grad_b = grad_B.pop()

                self.layer_list[index].W += learning_rate * grad_w
                self.layer_list[index].b += learning_rate * grad_b
                index += 1

            if _ % 100 == 0:
                loss = self.cost(self.X, tensor_T, C=C)
                losses.append(loss)
                Y = self.forward(self.X)
                print('Train Accuracy:', self.classification_rate(Y, self.T), "--loss:", loss)

        plt.plot(losses)
        plt.title("loss per iteration")
        plt.show()

        self.Y = self.forward(self.X)
        print('Train Accuracy:', self.classification_rate(self.Y, self.T))


'''------------Test Datas-------------'''


def get_xor():
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50, 2)) / 2 + 0.5  # (0.5-1, 0.5-1)
    X[50:100] = np.random.random((50, 2)) / 2  # (0-0.5, 0-0.5)
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]])  # (0-0.5, 0.5-1)
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]])  # (0.5-1, 0-0.5)
    Y = np.array([0] * 100 + [1] * 100)
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
    return X, Y


def get_cloud():
    # create the data
    Nclass = 500
    D = 2  # dimensionality of input

    X1 = np.random.randn(Nclass, D) + np.array([0, -2])
    X2 = np.random.randn(Nclass, D) + np.array([2, 2])
    X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])

    Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    X_train = np.asarray(X_train.todense())
    X_test = np.asarray(X_test.todense())

    return X_train, X_test, y_train, y_test


# Use the Neural Network to solve the problem
X, T = get_cloud()

model = ANN(X, T)
model.add_layer(hidden_size=5, activation=relu, derivative_activation=derivative_relu)
model.add_layer(hidden_size=3, activation=softmax)
model.backpropagation(n_iters=1000, learning_rate=1e-3, C=0.)
