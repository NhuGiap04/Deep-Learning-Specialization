import numpy as np
import matplotlib.pyplot as plt

N = 50

X = np.linspace(0, 10, N)
y = 0.5 * X + np.random.randn(N)


y[-10: -1] += 30

my_lambda = 10000.


def cost_function(y_output, y_predict, _lambda, weight):
    n = len(y_output)
    y_output = np.reshape(y_output, newshape=(n,))
    y_predict = np.reshape(y_predict, newshape=(n,))
    weight = np.reshape(weight, newshape=(len(weight),))
    squared_error = np.dot((y_output - y_predict).T, y_output - y_predict)
    return 1/n * squared_error + _lambda * np.dot(weight.T, weight)


def solve_regularize_weight(X_input, y_output, _lambda):
    if X_input.ndim == 1:
        X_input = np.reshape(X_input, newshape=(len(X_input), 1))
    X_input = np.concatenate((np.ones(shape=(len(y_output), 1)), X_input), axis=1)
    weight = np.linalg.solve(_lambda * np.identity(len(X_input[0])) + np.dot(X_input.T, X_input), np.dot(X_input.T, y_output))
    return weight


def solve_weights(X_input, y_output):
    if X_input.ndim == 1:
        X_input = np.reshape(X_input, newshape=(len(X_input), 1))
    X_input = np.concatenate((np.ones(shape=(len(y_output), 1)), X_input), axis=1)
    weight = np.linalg.solve(np.dot(X_input.T, X_input), np.dot(X_input.T, y_output))
    return weight


def compute_predict(X_input, weight, number_sample):
    X_input = np.concatenate((np.ones(shape=(number_sample, 1)), X_input), axis=1)
    return np.dot(X_input, weight)


def r_squared(y_output, y_predict, s):
    y_predict = np.reshape(y_predict, (len(y),))
    output_mean = np.mean(y_output)
    # Calculate the residual sum and the total sum
    sum_squared_residual = np.dot(np.subtract(y, y_predict).T, np.subtract(y, y_predict))
    sum_squared_total = np.dot((y - output_mean).T, y - output_mean)
    r_squared_value = 1 - sum_squared_residual / sum_squared_total
    print(f"R Squared in {s} model is: {r_squared_value}")


def plot_result(X_input, y_input, y_predict, s):
    plt.scatter(X_input, y_input, color='red')
    plt.plot(X_input, y_predict, color='blue')
    if s == "Regularized":
        plt.title('Regularized Linear Regression')
    elif s == "Normal":
        plt.title('Normal Linear Regression')
    plt.show()


if __name__ == "__main__":
    w = solve_weights(X, y)
    w_regularized = solve_regularize_weight(X, y, my_lambda)
    X = np.reshape(X, (len(X), 1))

    y_predict = compute_predict(X, w, len(y))
    plot_result(X, y, y_predict, "Normal")
    r_squared(y, y_predict, "Normal")

    y_regularized_predict = compute_predict(X, w_regularized, len(y))
    plot_result(X, y, y_regularized_predict, "Regularized")
    r_squared(y, y_regularized_predict, "Regularized")

