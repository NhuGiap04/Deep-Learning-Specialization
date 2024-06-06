import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
dataset = pd.read_csv("data_poly.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# poly_reg = PolynomialFeatures(degree=2)
# X_poly = poly_reg.fit_transform(X)
mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, size=(len(y), 1))
X_poly = np.concatenate((X, s), axis=1)

# Solve the coefficients(weights)
w = np.linalg.solve(np.dot(X_poly.T, X_poly), np.dot(X_poly.T, y))
y_predict = np.dot(X_poly, w)

# Plot the result
plt.scatter(X, y, color='red')
plt.show()


# Plot it all together
plt.scatter(X, y)
plt.plot(sorted(X), sorted(y_predict))
plt.show()

# R-squared
y_predict = np.reshape(y_predict, (len(y),))
output_mean = np.mean(y)
sum_squared_residual = np.dot(np.subtract(y, y_predict).T, np.subtract(y, y_predict))
sum_squared_total = np.dot((y - output_mean).T, y - output_mean)
r_squared_value = 1 - sum_squared_residual/sum_squared_total
print(f"R Squared in this model is: {r_squared_value}")
