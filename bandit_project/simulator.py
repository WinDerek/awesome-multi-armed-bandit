import numpy as np
from machine import get_reward
import matplotlib.pyplot as plt


class Simulator:
    def __init__(self, number_of_time_slot=10000):
        self.number_of_time_slot = number_of_time_slot
        self.reward_ndarray = None
        self.aggregated_reward_ndarray = None
        self.average_reward_ndarray = None
        self.agent_name_array = None


    def run_agent(self, agent):
        # Reset the agent
        agent.reset()

        # Initialization
        reward_array = np.zeros(self.number_of_time_slot, dtype=float)
        aggregated_reward_array = np.zeros(self.number_of_time_slot, dtype=float)
        average_reward_array = np.zeros(self.number_of_time_slot, dtype=float)

        # Let the agent pull arms repeatedly
        for t in range(1, self.number_of_time_slot + 1):
            I, reward = agent.pull_arm()

            reward_array[t-1] = reward
            aggregated_reward_array[t-1] = np.sum(reward_array[0:t])
            average_reward_array[t-1] = np.average(reward_array[0:t])
        
        # Store the reward
        if self.reward_ndarray is None:
            self.reward_ndarray = reward_array
        else:
            self.reward_ndarray = np.vstack((self.reward_ndarray, reward_array))

        # Store the aggregated reward
        if self.aggregated_reward_ndarray is None:
            self.aggregated_reward_ndarray = aggregated_reward_array
        else:
            self.aggregated_reward_ndarray = np.vstack((self.aggregated_reward_ndarray, aggregated_reward_array))

        # Store the average reward
        if self.average_reward_ndarray is None:
            self.average_reward_ndarray = average_reward_array
        else:
            self.average_reward_ndarray = np.vstack((self.average_reward_ndarray, average_reward_array))

        # Store the name of the agent
        if self.agent_name_array is None:
            self.agent_name_array = np.array([agent.name()])
        else:
            self.agent_name_array = np.append(self.agent_name_array, agent.name())


    def plot(self):
        color_list = ['#cf000f', '#19b5fe', '#29f1c3']

        fig, ax = plt.subplots(figsize=(16,8), dpi=200)
        for i in range(self.reward_ndarray.shape[0]):
            ax.plot(range(1, self.number_of_time_slot + 1), self.aggregated_reward_ndarray[i], '--', color=color_list[i], label=self.agent_name_array[i])
            ax.legend(loc='lower right')
            ax.set_title('Aggregated reward v.s. time slot $t$')
            ax.set_xlabel('Time slot $t$')
            ax.set_ylabel('Aggregated reward')

        fig, ax = plt.subplots(figsize=(16,8), dpi=200)
        for i in range(self.reward_ndarray.shape[0]):
            ax.plot(range(1, self.number_of_time_slot + 1), self.average_reward_ndarray[i], '--', color=color_list[i], label=self.agent_name_array[i])
            ax.legend(loc='lower right')

        fig, ax = plt.subplots(figsize=(16,8), dpi=200)
        for i in range(self.reward_ndarray.shape[0]):
            ax.plot(range(1, 301), self.average_reward_ndarray[i][0:300], '--', color=color_list[i], label=self.agent_name_array[i])
            ax.legend(loc='lower right')

        plt.show()
