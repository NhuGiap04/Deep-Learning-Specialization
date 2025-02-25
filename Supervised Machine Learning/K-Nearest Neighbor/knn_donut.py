from k_nearest_neighbor import KNN
from util import get_donut
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X, Y = get_donut()

    # display the data
    plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
    plt.show()

    # get the accuracy
    model = KNN(3)
    model.fit(X, Y)
    print("Accuracy:", model.score(X, Y))
