import numpy as np
from agent import Agent
from machine import get_reward


class EpsilonGreedyAgent(Agent):
    def __init__(self, epsilon, **kwds):
        self.epsilon = epsilon

        self.reset()

        super().__init__(**kwds)


    def reset(self):
        self.theta_hat_array = np.zeros(3, dtype=float)
        self.count_array = np.zeros(3, dtype=int)

    
    def pull_arm(self):
        I = -1
        # Choose I randomly with probability epsilon
        if np.random.binomial(1, self.epsilon):
            I = np.random.randint(1, 4)
        # Get greedy with probability (1 - epsilon)
        else:
            I = np.argmax(self.theta_hat_array) + 1
        reward = get_reward(I)

        return I, reward


    def name(self):
        return '$\epsilon$-greedy with $\epsilon = {}$'.format(self.epsilon)
