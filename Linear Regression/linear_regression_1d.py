import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
'''Method 1'''
# dataset = pd.read_csv('data_1d.csv')
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, -1].values
'''Method 2'''
X = []
y = []
non_decimal = re.compile(r'[^\d]+')
for line in open('moore.csv'):
    r = line.split('\t')

    x = int(non_decimal.sub('', r[2].split('[')[0]))
    _y = int(non_decimal.sub('', r[1].split('[')[0]))
    X.append(x)
    y.append(_y)
X = np.array(X)
X = np.log2(X)
y = np.array(y)
y = np.log2(y)


# Calculate the coefficients
sample_mean = np.mean(X)
output_mean = np.mean(y)
xy_mean = np.dot(X.T, y) / len(y)
squared_mean = np.dot(X.T, X) / len(X)

a = (xy_mean - sample_mean * output_mean) / (squared_mean - sample_mean ** 2)
b = (output_mean * squared_mean - sample_mean * xy_mean) / (squared_mean - sample_mean ** 2)

# Plot the result
y_predict = a * X + b
plt.scatter(X, y, color='red')
plt.plot(X, y_predict, color='blue')
plt.title('Simple Linear Regression')
plt.xlabel('Time (In Years)')
plt.ylabel('Transistor Count')
plt.show()

# R-squared
y_predict = np.reshape(y_predict, (len(y),))
sum_squared_residual = np.dot(np.subtract(y, y_predict).T, np.subtract(y, y_predict))
sum_squared_total = np.dot((y - output_mean).T, y - output_mean)
r_squared_value = 1 - sum_squared_residual/sum_squared_total
print(f"R Squared in this model is: {r_squared_value}")
