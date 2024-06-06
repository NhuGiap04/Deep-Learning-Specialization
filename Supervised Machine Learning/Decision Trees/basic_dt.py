from util import get_xor, get_donut, get_data
from datetime import datetime
import numpy as np

from sklearn.model_selection import train_test_split


class TreeNode:
    def __init__(self):
        self.left_node = None
        self.right_node = None
        self.left_prediction = None
        self.right_prediction = None
        # Attribute to make Prediction
        self.condition = None
        self.attribute = None

    def entropy(self, y):
        N = len(y)
        dict_y = dict()
        for i in y:
            if i not in dict_y.keys():
                dict_y[i] = 1
            else:
                dict_y[i] += 1
        E = 0
        for value in dict_y.values():
            E += -np.log2(value/N) * value/N
        return E

    def find_split(self, X, Y, c):
        N = len(Y)
        # Find split for X[c] and Y
        X_c = X[:, c].reshape((len(X[:, c]), 1))
        Y_c = Y.reshape((len(Y), 1))
        sub_data = np.concatenate((X_c, Y_c), axis=1)

        # Get the sorted data by sorting X[c]
        sort_data = sub_data[sub_data[:, 0].argsort()]

        best_split = (sort_data[0][0] + sort_data[1][0]) / 2
        best_IG = 0

        init_entropy = self.entropy(Y)
        for index in range(len(sort_data) - 1):
            if sort_data[index][1] != sort_data[index + 1][1]:
                info_gain = init_entropy - index/N * self.entropy([Y[i] for i in range(index + 1)]) - (N-index)/N * self.entropy([Y[i] for i in range(index + 1, len(sort_data))])
                if info_gain > best_IG:
                    best_IG = info_gain
                    best_split = (sort_data[index][0] + sort_data[index + 1][0]) / 2
        return best_split

    def split(self, X, Y, c, condition):
        X_left = X[X[:, c] <= condition]
        Y_left = Y[X[:, c] <= condition]
        X_right = X[X[:, c] > condition]
        Y_right = Y[X[:, c] > condition]

        return X_left, Y_left, X_right, Y_right

    def information_gain(self, X, Y, column, condition):
        Y_left = Y[X[:, column] <= condition]
        Y_right = Y[X[:, column] > condition]
        info_gain = self.entropy(Y) - len(Y_left) / len(Y) * self.entropy(Y_left) - len(Y_right) / len(
            Y) * self.entropy(Y_right)

        return info_gain

    def fit(self, X, Y, max_depth=20):
        best_IG = -1
        best_attribute = 0

        for c in range(len(X[0])):
            condition = self.find_split(X, Y, c)
            info_gain = self.information_gain(X, Y, c, condition)
            if info_gain > best_IG:
                best_IG = info_gain
                best_attribute = c
                # Set the attributes
                self.condition = condition
                self.attribute = best_attribute

        X_left, Y_left, X_right, Y_right = self.split(X, Y, best_attribute, self.condition)
        if best_IG == 0:  # First base case: Max Information Gain = 0
            self.left_prediction = np.bincount(Y_left).argmax()
            self.right_prediction = np.bincount(Y_right).argmax()
        elif max_depth == 0:  # Second base case: Max Depth reaches 0
            self.left_prediction = np.bincount(Y_left).argmax()
            self.right_prediction = np.bincount(Y_right).argmax()
        else:
            if len(Y_left) == 1 or np.all(Y_left == Y_left[0]):
                self.left_prediction = np.bincount(Y_left).argmax()
            else:
                self.left_node = TreeNode()
                self.left_node.fit(X_left, Y_left, max_depth - 1)
            if len(Y_right) == 1 or np.all(Y_right == Y_right[0]):
                self.right_prediction = np.bincount(Y_right).argmax()
            else:
                self.right_node = TreeNode()
                self.right_node.fit(X_right, Y_right, max_depth - 1)

    def predict_one(self, x):
        if x[self.attribute] <= self.condition:
            if self.left_node is not None:
                return self.left_node.predict_one(x)
            else:
                return self.left_prediction
        else:
            if self.right_node:
                return self.right_node.predict_one(x)
            else:
                return self.right_prediction

    def score(self, X, Y):
        Y_predict = []
        for test in X:
            Y_predict.append(self.predict_one(test))
        Y_predict = np.array(Y_predict)
        return np.sum(Y_predict == Y) / len(Y)


'''----------------------IMPLEMENTATION----------------------'''
model = TreeNode()

X, Y = get_donut()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

t0 = datetime.now()
model.fit(X_train, Y_train, max_depth=30)
print("Training time:", (datetime.now() - t0))

t0 = datetime.now()
print("Train accuracy:", model.score(X_train, Y_train))
print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Y_train))


t0 = datetime.now()
print("Test accuracy:", model.score(X_test, Y_test))
print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Y_test))
