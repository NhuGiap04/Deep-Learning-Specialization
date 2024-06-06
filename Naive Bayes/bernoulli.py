# Implement BernoulliNB in scikit-learn
# create a class Bernoulli and the method fit, predict, score just like in Scikit-learn

# Import the libraries
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from scipy.stats import bernoulli as bn


# Define the Gaussian class
class BernoulliNB:
    def __init__(self, smoothing=1e-3):
        self.smoothing = smoothing

    def fit(self, X, y):
        pass

    def score(self, X, y):
        pass

    def predict(self, X):
        pass


# Import the dataset
df = pd.read_csv('dna.csv')
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Preprocess the data (convert it to the suitable shape)
X_train = df_train.iloc[:, :-1].values
y_train = df_train.iloc[:, -1].values

X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values

# Test the model
model = BernoulliNB(smoothing=1e-2)

t0 = datetime.now()
model.fit(X_train, y_train)
print("fit duration:", datetime.now() - t0)

t0 = datetime.now()
print("train accuracy:", model.score(X_train, y_train))
print("train predicting duration:", datetime.now() - t0)

t0 = datetime.now()
print("test accuracy:", model.score(X_test, y_test))
print("test predicting duration:", datetime.now() - t0)