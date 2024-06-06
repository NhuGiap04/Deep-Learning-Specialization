# Import the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Get the data
df = pd.read_csv('Concrete_Data.csv')
df.columns = list(range(df.shape[1]))

X = df[[0, 1, 2, 3, 4, 5, 6, 7]].values
y = df[8].values

# split the data into train and test sets
# this lets us simulate how our model will perform in the future
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVR(kernel='rbf')

t0 = datetime.now()
model.fit(X_train, y_train)
print('train duration:', datetime.now() - t0)

t0 = datetime.now()
# R-squared score since this is a regression model
print('train score:', model.score(X_train, y_train), 'duration:', datetime.now() - t0)

t0 = datetime.now()
print('test score:', model.score(X_test, y_test), 'duration:', datetime.now() - t0)
