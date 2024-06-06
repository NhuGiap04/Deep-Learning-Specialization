# Import the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# load the data
data = load_breast_cancer()

for C in (0.5, 1.0, 5.0, 10.0):
    # Pipeline: Scikit-learn automatically know fitting should only be done on the Train data and not on the Test data
    pipeline = Pipeline([('scaler', StandardScaler()), ('svm', SVC(C=C))])
    # cv: number of folds
    scores = cross_val_score(pipeline, data.data, data.target, cv=5)
    print("C:", C, "mean:", scores.mean(), "std:", scores.std())
