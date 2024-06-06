import matplotlib.pyplot as plt
import numpy as np
import random
import math

NUM_TRIALS = 10000
'''Constant Epsilon'''
# EPS = 0.1
INIT_EPS = 1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0
        self.N = 0

    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p
        # np.random.random() Return random floats in the half-open interval [0.0, 1.0)

    def update(self, x):
        self.N += 1
        self.p_estimate = self.p_estimate + 1/self.N * (x - self.p_estimate)


def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    rewards = np.zeros(NUM_TRIALS)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0
    optimal_j = np.argmax([b.p for b in bandits])
    print("optimal j: ", optimal_j)

    for i in range(NUM_TRIALS):
        EPS = max(INIT_EPS - 0.001 * i, 0.001)
        # use epsilon-greedy to select the next bandits
        if np.random.random() < EPS:
            num_times_explored += 1
            j = random.randint(0, len(bandits) - 1)
        else:
            num_times_exploited += 1
            j = np.argmax([b.p_estimate for b in bandits])

        if j == optimal_j:
            num_optimal += 1

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        # update rewards log
        rewards[i] = x

        # update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

    # print mean estimates for each bandit
    i = 1
    for b in bandits:
        print(f"mean bandit {i} estimate:", b.p_estimate)
        i += 1

    # print total reward
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num_times_explored:", num_times_explored)
    print("num_times_exploited:", num_times_exploited)
    print("num times selected optimal bandit:", num_optimal)

    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.title('Epsilon Greedy')
    plt.xlabel('Number of Trials')
    plt.ylabel('Probabilities')
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))
    plt.show()


if __name__ == "__main__":
    experiment()
