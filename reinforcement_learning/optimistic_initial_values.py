# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt
from reinforcement_learning.basic_bandit import DefaultBandit


class Bandit(DefaultBandit):

    def __init__(self, m, upper_limit):
        super().__init__(m)
        self.mean = upper_limit

def run_experiment(m1, m2, m3, N):
    bandits = [Bandit(m1, 10), Bandit(m2, 10), Bandit(m3, 10)]

    data = np.empty(N)

    for i in range(N):
        # epsilon greedy
        j = np.argmax([b.mean for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)

        # for the plot
        data[i] = x
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

    # plot moving average ctr
    plt.plot(cumulative_average)
    plt.plot(np.ones(N) * m1)
    plt.plot(np.ones(N) * m2)
    plt.plot(np.ones(N) * m3)
    plt.xscale('log')
    plt.show()

    for b in bandits:
        print(b.mean)

    return cumulative_average


if __name__ == '__main__':
    c_1 = run_experiment(1.0, 2.0, 3.0, 100000)

    # log scale plot
    plt.plot(c_1, label='optimistic = 10')
    plt.legend()
    plt.xscale('log')
    plt.show()

    # linear plot
    plt.plot(c_1, label='optimistic = 10')
    plt.legend()
    plt.show()

