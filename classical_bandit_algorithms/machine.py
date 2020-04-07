import numpy as np


# The true values of the parameters of the reward distributions
theta = [0.4, 0.6, 0.8]

def get_reward(i):
    """
    Returns the reward for the arm `i`.

    Args:
        i (int): The index of the arm. Acceptable values: `1`, `2` and `3`.

    Returns:
        The reward for the arm `i` when the value of `i` is acceptable, `-1` otherwise.
    """

    if i not in range(1, 4):
        return -1

    return np.random.binomial(1, theta[i-1])
