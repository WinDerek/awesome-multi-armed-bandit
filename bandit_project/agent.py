import numpy as np
from machine import get_reward


class Agent:
    def __init__(self):
        pass


    def reset(self):
        pass


    def pull_arm(self):
        """
        Returns:
            (arm_index, reward)
        """
        return -1, -1.0


    def name(self):
        return 'agent'


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

        self.count_array[I-1] += 1
        self.theta_hat_array[I-1] += (reward - self.theta_hat_array[I-1]) / self.count_array[I-1]

        return I, reward


    def name(self):
        return '$\epsilon$-greedy with $\epsilon = {}$'.format(self.epsilon)
