import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


NUM_TRIALS = 10000
BANDIT_PROBABILITIES = [(0.2, 1), (0.5, 1), (0.75, 1)]


class Bandit:
    def __init__(self, mean, standard_deviation):
        # Thompson sampling for unknown mean with known precision
        self.local_mean = 0
        self.m = 0
        self.lamda = 1
        self.sum_x = 0
        # Define the true Gaussian distribution
        self.mean = float(mean)
        self.standard_deviation = float(standard_deviation)
        self.precision = 1 / (self.standard_deviation ** 2)
        self.N = 0

    def pull(self):
        # draw a 1 with Gaussian distribution(mean, std)
        return np.random.normal(loc=self.mean, scale=self.standard_deviation)

    def update(self, x):
        self.N += 1
        self.lamda += self.precision
        self.sum_x += x
        self.m = (self.precision * self.sum_x) / self.lamda

    def sample(self):
        return np.random.normal(float(self.m), np.sqrt(float(1 / self.lamda)))


def plot(bandits, trial):
    x = np.linspace(0, 1, 200)
    for b in bandits:
        y = norm.pdf(x, b.m, np.sqrt(1/b.lamda))
        plt.plot(x, y, label=f"true mean: {b.mean:.4f}, num plays: {b.N}")
    plt.title(f"Bandit distributions after {trial} trials")
    plt.legend()
    plt.show()


def experiment():
    bandits = [Bandit(a, b) for (a, b) in BANDIT_PROBABILITIES]

    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]
    rewards = np.zeros(NUM_TRIALS)

    num_optimal = 0
    optimal_j = np.argmax([b.mean for b in bandits])
    print("optimal j: ", optimal_j)

    for i in range(NUM_TRIALS):
        j = np.argmax([b.sample() for b in bandits])

        if j == optimal_j:
            num_optimal += 1

        if i in sample_points:
            plot(bandits, i)

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        # update rewards log
        rewards[i] = x

        # update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

    # print total reward
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num times selected optimal bandit:", num_optimal)

    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.title('Thompson Sampling with Gaussian Reward')
    plt.xlabel('Number of Trials')
    plt.ylabel('Probabilities')
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS) * np.max([BANDIT_PROBABILITIES[i][0] for i in range(len(BANDIT_PROBABILITIES))]))
    plt.show()


if __name__ == "__main__":
    experiment()