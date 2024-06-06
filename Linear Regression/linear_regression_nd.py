import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
dataset = pd.read_csv("data_2d.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X = np.concatenate((np.ones(shape=(len(y), 1)), X), axis=1)

# Solve the coefficients(weights)
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))

# Plot the result
y_predict = np.dot(X, w)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y)
plt.show()

# Compute the R-squared
y_predict = np.reshape(y_predict, (len(y),))
output_mean = np.mean(y)
sum_squared_residual = np.dot(np.subtract(y, y_predict).T, np.subtract(y, y_predict))
sum_squared_total = np.dot((y - output_mean).T, y - output_mean)
r_squared_value = 1 - sum_squared_residual/sum_squared_total
print(r_squared_value)

