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


class SoftKMean:
    def __init__(self):
        return

    def get_means(self, X, R, step=1):
        # R is the responsibility matrix (N x K)
        K, D = R.shape[1], X.shape[1]
        M = np.zeros((K, D))
        for index in range(K):
            M[index] = R[:, index].dot(X) / np.sum(R[:, index])
        return M

    def get_res_mat(self, X, M, beta=3.0):
        N = X.shape[0]
        K = M.shape[0]
        distance_mat = np.zeros((N, K))
        for index in range(K):
            dist_vect = np.sum((X - M[index]) * (X - M[index]), axis=1)  # No need to square-root the term
            distance_mat[:, index] = dist_vect
        R = np.exp(distance_mat * -beta)  # shape: NxK
        for index in range(N):
            R[index] = R[index] / np.sum(R[index])
        return R

    def get_simple_plot(self, X, R):
        if X.shape[1] != 2:
            return
        N, K = R.shape
        Y = np.argmax(R, axis=1)
        # Depicting the Visualization
        color_list = ['blue', 'green', 'red', 'orange', 'purple', 'pink', 'brown']
        for index in range(K):
            plt.scatter(X[Y == index][:, 0], X[Y == index][:, 1], color=color_list[index])
        plt.xlabel('X1')
        plt.ylabel('X2')
        mean = self.get_means(X, R)
        plt.scatter(mean[:, 0], mean[:, 1], color=color_list[K])

        # Displaying the title
        plt.title(label="K-Means Clustering", fontsize=20)
        plt.show()

    def objective_func(self, X, M, R):
        J = 0
        N, K = R.shape
        for index in range(K):
            T = np.sum((X - M[index]) * (X - M[index]), axis=1)
            J += R[:, index].dot(T)
        return J

    def fit(self, X, K, beta=3.0, max_iters=20, show_plots=False):
        rand_ind = np.random.permutation(len(X))
        curr_centers = X[rand_ind[:K]]
        step = 1
        costs = []
        while True:
            R = self.get_res_mat(X, curr_centers, beta=beta)
            new_centers = self.get_means(X, R, step)
            if np.array_equal(curr_centers, new_centers):
                if show_plots:
                    print(f"Converges at step {step}")
                break
            elif max_iters < step:
                break
            else:
                curr_centers = new_centers
            step += 1
            costs.append(self.objective_func(X, curr_centers, R))
        if show_plots:
            self.get_simple_plot(X, R)
            plt.plot(costs)
            plt.show()
        return curr_centers, R


'''----------------------IMPLEMENTATION----------------------'''
model = SoftKMean()

X = get_data()

t0 = datetime.now()
# Sometimes the algorithm may be stuck since beta is too small
model.fit(X, 3, beta=3.0, max_iters=100)
print("Training time:", (datetime.now() - t0))
