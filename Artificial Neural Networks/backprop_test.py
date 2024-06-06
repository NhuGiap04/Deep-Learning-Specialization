import tensorflow as tf
import numpy as np


def get_rand_data():
    X = np.array([[1., 2.],
                  [3., 4.],
                  [5., 6.]])
    T = np.array([[1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]])
    return X, T


X, T = get_rand_data()
X = tf.convert_to_tensor(X,
                         dtype=tf.float32)
T = tf.convert_to_tensor(T,
                         dtype=tf.float32)
W1 = tf.Variable([[-0.05427205, -1., -0.49127007],
                  [0.34459662, -0.05660348, -0.49049374]], dtype=tf.float32)
b1 = tf.Variable([[-1., -0.15455505, -0.5138969]], dtype=tf.float32)
W2 = tf.Variable([[0.10485189, -1., 0.62274504],
                  [0.73552126, -0.31967974, 0.28646442],
                  [0.5224993, -0.4967681, 0.0554093]], dtype=tf.float32)
b2 = tf.Variable([[-1., -0.5534871, -0.0077894]], dtype=tf.float32)

Z1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
Z2 = tf.nn.softmax(tf.matmul(Z1, W2) + b2)

delta2 = T - Z2
print(W2 - 1e-3 * tf.matmul(tf.transpose(Z1), T - Z2))
print(b2 - 1e-3 * tf.reduce_sum(delta2, axis=0))
delta1 = tf.matmul(delta2, tf.transpose(W2)) * (Z1 * (1 - Z1))
print(W1 - 1e-3 * tf.matmul(tf.transpose(X), delta1))
print(b1 - 1e-3 * tf.reduce_sum(delta1, axis=0))

