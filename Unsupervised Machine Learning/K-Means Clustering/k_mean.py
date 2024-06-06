import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


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


class KMeansCluster:
    def __init__(self):
        return

    def get_means(self, X, Y, K):
        D = len(X[0])  # The number of features
        M = np.zeros((K, D))
        for index in range(K):
            index_sample = X[Y == index]
            M[index] = np.mean(index_sample, axis=0)
        return M

    def get_cluster(self, X, M):
        N = len(X)
        K, D = M.shape
        Y = np.zeros((N, ))
        # Set up the minimum Euclidean-distance matrix
        T = np.zeros((N, )) + float('inf')
        for index in range(K):
            distance_vect = np.sum((X - M[index]) * (X - M[index]), axis=1)
            Y[T > distance_vect] = index
            T = np.minimum(T, distance_vect)
        return Y

    def get_simple_plot(self, X, Y, K):
        if X.shape[1] != 2:
            return
        # Depicting the Visualization
        color_list = ['blue', 'green', 'red', 'orange', 'purple', 'pink', 'brown']
        for index in range(K):
            plt.scatter(X[Y == index][:, 0], X[Y == index][:, 1], color=color_list[index])
        plt.xlabel('X1')
        plt.ylabel('X2')
        mean = self.get_means(X, Y, K)
        plt.scatter(mean[:, 0], mean[:, 1], color=color_list[K])

        # Displaying the title
        plt.title(label="K-Means Clustering", fontsize=20)
        plt.show()

    def objective_func(self, X, Y, M):
        K = M.shape[0]
        J = 0
        for index in range(K):
            J += np.sum((X[Y == index] - M[index]) * (X[Y == index] - M[index]))
        return J

    def fit(self, X, K):
        rand_ind = np.random.permutation(len(X))
        curr_centers = X[rand_ind[:K]]
        step = 0
        costs = []
        while True:
            Y = self.get_cluster(X, curr_centers)
            new_centers = self.get_means(X, Y, K)
            if np.array_equal(curr_centers, new_centers):
                print(f"Converges at step {step}")
                break
            else:
                curr_centers = new_centers
                costs.append(self.objective_func(X, Y, new_centers))
            step += 1
        self.get_simple_plot(X, Y, K)
        plt.plot(costs)
        plt.show()


'''----------------------IMPLEMENTATION----------------------'''
model = KMeansCluster()

X = get_data()

t0 = datetime.now()
model.fit(X, 3)
print("Training time:", (datetime.now() - t0))


