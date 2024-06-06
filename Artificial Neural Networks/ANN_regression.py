import numpy as np
import matplotlib.pyplot as plt

# generate and plot the data
N = 500
X = np.random.random((N, 2)) * 4 - 2  # in between (-2, +2)
Y = X[:, 0] * X[:, 1]  # makes a saddle shape
# note: in this script "Y" will be the target,
#       "Yhat" will be prediction

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()

# make a neural network and train it
D = 2
M = 100  # number of hidden units

# layer 1
W = np.random.randn(D, M) / np.sqrt(D)
b = np.zeros(M)

# layer 2
V = np.random.randn(M) / np.sqrt(M)
c = 0


# how to get the output
# consider the params global
def forward(X):
    pass


# how to train the params
def derivative_V(Z, Y, Yhat):
    pass


def derivative_c(Y, Yhat):
    pass


def derivative_W(X, Z, Y, Yhat, V):
    pass


def derivative_b(Z, Y, Yhat, V):
    pass


def update(X, Z, Y, Yhat, W, b, V, c, learning_rate=1e-4):
    pass


# so we can plot the costs later
def get_cost(Y, Yhat):
    pass


# run a training loop
# plot the costs
# and plot the final result
costs = []
for i in range(200):
    Z, Yhat = forward(X)
    W, b, V, c = update(X, Z, Y, Yhat, W, b, V, c)
    cost = get_cost(Y, Yhat)
    costs.append(cost)
    if i % 25 == 0:
        print(cost)

# plot the costs
plt.plot(costs)
plt.show()

# plot the prediction with the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)

# surface plot
line = np.linspace(-2, 2, 20)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
_, Yhat = forward(Xgrid)
ax.plot_trisurf(Xgrid[:, 0], Xgrid[:, 1], Yhat, linewidth=0.2, antialiased=True)
plt.show()

# plot magnitude of residuals
Ygrid = Xgrid[:, 0] * Xgrid[:, 1]
R = np.abs(Ygrid - Yhat)

plt.scatter(Xgrid[:, 0], Xgrid[:, 1], c=R)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(Xgrid[:, 0], Xgrid[:, 1], R, linewidth=0.2, antialiased=True)
plt.show()
