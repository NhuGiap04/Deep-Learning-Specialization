import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage


def get_data():
    # Set up the parameters
    D = 2
    N = 300  # Number of points in each cluster
    s = 2  # separation so we can control how far apart the means are
    mu_1 = np.array([0, 0])
    mu_2 = np.array([s, s])
    mu_3 = np.array([0, s])
    # Generate the data
    X = np.zeros((3 * N, D))
    X[:300, :] = np.random.random((N, D)) + mu_1
    X[300:600, :] = np.random.random((N, D)) + mu_2
    X[600:, :] = np.random.random((N, D)) + mu_3

    return X


def main():
    X = get_data()

    Z = linkage(X, 'ward')
    print("Z.shape", Z.shape)

    plt.title("Ward")
    dendrogram(Z)
    plt.show()

    Z = linkage(X, 'single')
    plt.title("Single")
    dendrogram(Z)
    plt.show()

    Z = linkage(X, 'complete')
    plt.title("Complete")
    dendrogram(Z)
    plt.show()


if __name__ == "__main__":
    main()