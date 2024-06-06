import matplotlib.pyplot as plt
from soft_k_means import get_data, SoftKMean


def main():
    model = SoftKMean()
    X = get_data()

    costs = []
    for k in range(1, 9):
        M, R = model.fit(X, k, beta=5.0)
        costs.append(model.objective_func(X, M, R))

    plt.plot(costs)
    plt.title("Cost vs K")
    plt.show()


if __name__ == "__main__":
    main()
