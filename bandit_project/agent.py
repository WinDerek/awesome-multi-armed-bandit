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


class UcbAgent:
    def __init__(self, c, **kwds):
        self.c = c

        self.reset()

        super().__init__(**kwds)

    
    def reset(self):
        self.theta_hat_array = np.zeros(3, dtype=float)
        self.count_array = np.zeros(3, dtype=int)
        self.time_slot = 0

    
    def pull_arm(self):
        # Increment the time slot
        self.time_slot += 1

        # Select an arm
        I = -1
        if self.time_slot in range(1, 4):
            I = self.time_slot
        else:
            I = np.argmax([self.theta_hat_array[j] + self.c * np.sqrt(2 * np.log(self.time_slot) / self.count_array[j]) for j in range(3)]) + 1
        
        # Get the reward
        reward = get_reward(I)

        self.count_array[I - 1] += 1
        self.theta_hat_array[I - 1] += (reward - self.theta_hat_array[I - 1]) / self.count_array[I - 1]

        return I, reward

    
    def name(self):
        return 'UCB with c = {}'.format(self.c)


class TsAgent:
    def __init__(self, alpha, beta, **kwds):
        self.alpha = alpha
        self.beta = beta

        self.reset()

        super().__init__(**kwds)

    
    def reset(self):
        return


    def pull_arm(self):
        I = -1
        reward = -1
        return I, reward

    
    def name(self):
        return 'TS with $\alpha = {}$, $\beta = {}$'.format(self.alpha, self.beta)
