import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
# from gmm import gmm
from sklearn.mixture import GaussianMixture
from util import get_data, purity, DBI


def main():
    X, Y = get_data(10000)
    print("Number of data points:", len(Y))

    model = GaussianMixture(n_components=10)
    model.fit(X)
    M = model.means_
    R = model.predict_proba(X)

    print("Purity:", purity(Y, R))  # max is 1, higher is better
    print("DBI:", DBI(X, M, R))  # lower is better


if __name__ == "__main__":
    main()
