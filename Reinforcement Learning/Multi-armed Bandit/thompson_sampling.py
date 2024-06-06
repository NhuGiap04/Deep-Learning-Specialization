import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta


NUM_TRIALS = 10000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
    def __init__(self, p):
        self.a = 1
        self.b = 1
        self.p = p
        self.N = 0
        pass

    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p
        # np.random.random() Return random floats in the half-open interval [0.0, 1.0)

    def update(self, x):
        self.N += 1
        self.a += x
        self.b += 1 - x

    def sample(self):
        return np.random.beta(float(self.a), float(self.b))


def plot(bandits, trial):
    x = np.linspace(0, 1, 200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x, y, label=f"real p: {b.p:.4f}, win rate = {b.a - 1}/{b.N}")
    plt.title(f"Bandit distributions after {trial} trials")
    plt.legend()
    plt.show()


def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]
    rewards = np.zeros(NUM_TRIALS)

    num_optimal = 0
    optimal_j = np.argmax([b.p for b in bandits])
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
    plt.title('Thompson Sampling')
    plt.xlabel('Number of Trials')
    plt.ylabel('Probabilities')
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))
    plt.show()


if __name__ == "__main__":
    experiment()