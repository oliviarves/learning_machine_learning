from reinforcement_learning.comparing_epsilons import run_experiment as run_experiment_epsilon
from reinforcement_learning.optimistic_initial_values import run_experiment as run_experiment_optimistic
from reinforcement_learning.ucb1 import run_experiment as run_experiment_ucb1
import matplotlib.pyplot as plt

if __name__ == '__main__':
    c_opt = run_experiment_optimistic(1.0, 2.0, 3.0, 100000)
    c_1 = run_experiment_epsilon(1.0, 2.0, 3.0, 0.1, 100000)
    c_05 = run_experiment_epsilon(1.0, 2.0, 3.0, 0.5, 100000)
    c_01 = run_experiment_epsilon(1.0, 2.0, 3.0, 0.01, 100000)
    c_ucb1 = run_experiment_ucb1(1.0, 2.0, 3.0, 100000)

    # log scale plot
    plt.plot(c_opt, label='optimistic = 10')
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.plot(c_ucb1, label='ucb1')

    plt.legend()
    plt.xscale('log')
    plt.show()

    # linear plot
    plt.plot(c_opt, label='optimistic = 10')
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.plot(c_ucb1, label='ucb1')
    plt.legend()
    plt.show()

