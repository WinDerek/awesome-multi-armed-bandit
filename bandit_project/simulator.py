import numpy as np
from machine import get_reward
import matplotlib.pyplot as plt


class Simulator:
    def __init__(self, number_of_time_slot=10000, number_of_repetition=100):
        self.number_of_time_slot = number_of_time_slot
        self.number_of_repetition = number_of_repetition

        self.reward_3d_ndarray = np.zeros((0, self.number_of_repetition, self.number_of_time_slot), dtype=float)
        self.mean_reward_2d_ndarray = np.zeros((0, self.number_of_time_slot), dtype=float)
        self.aggregated_reward_3d_ndarray = np.zeros((0, self.number_of_repetition, self.number_of_time_slot), dtype=float)
        self.mean_aggregated_reward_2d_ndarray = np.zeros((0, self.number_of_time_slot), dtype=float)
        self.average_reward_3d_ndarray = np.zeros((0, self.number_of_repetition, self.number_of_time_slot), dtype=float)
        self.mean_average_reward_2d_ndarray = np.zeros((0, self.number_of_time_slot), dtype=float)
        self.agent_name_array = np.array([])


    def run_agent(self, agent):
        reward_2d_ndarray = np.zeros((self.number_of_repetition, self.number_of_time_slot), dtype=float)
        aggregated_reward_2d_ndarray = np.zeros((self.number_of_repetition, self.number_of_time_slot), dtype=float)
        average_reward_2d_ndarray = np.zeros((self.number_of_repetition, self.number_of_time_slot), dtype=float)

        for i in range(self.number_of_repetition):
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
            
            # Store the reward array for this iteration in the 2d ndarray for reward
            reward_2d_ndarray[i] = reward_array

            # Store the aggregated reward array for this iteration in the 2d ndarray for aggregated reward
            aggregated_reward_2d_ndarray[i] = aggregated_reward_array

            # Store the average reward array for this iteration in the 2d ndarray for average reward
            average_reward_2d_ndarray[i] = average_reward_array
        
        # Store the data for reward
        self.__store_data(self.reward_3d_ndarray, self.mean_reward_2d_ndarray, reward_2d_ndarray)

        # Store the data for aggregated rewrad
        self.__store_data(self.aggregated_reward_3d_ndarray, self.mean_aggregated_reward_2d_ndarray, aggregated_reward_2d_ndarray)
        
        # Store the data for average reward
        self.__store_data(self.average_reward_3d_ndarray, self.mean_average_reward_2d_ndarray, average_reward_2d_ndarray)

        # Store the name of the agent
        self.agent_name_array = np.append(self.agent_name_array, agent.name())

        # Print information of executation
        print('Agent \'{}\' completed.'.format(agent.name()))


    def plot(self):
        color_list = ['#cf000f', '#19b5fe', '#29f1c3']

        number_of_agent = self.reward_3d_ndarray.shape[0]

        # Plot the mean aggregated reward
        fig, ax = plt.subplots(figsize=(16,8), dpi=200)
        for i in range(number_of_agent):
            ax.plot(range(1, self.number_of_time_slot + 1), self.mean_aggregated_reward_2d_ndarray[i], '--', color=color_list[i], label=self.agent_name_array[i])
            ax.legend(loc='lower right')
            ax.set_title('Mean aggregated reward v.s. time slot $t$')
            ax.set_xlabel('Time slot $t$')
            ax.set_ylabel('Mean aggregated reward')

        # Plot the mean average reward
        fig, ax = plt.subplots(figsize=(16,8), dpi=200)
        for i in range(number_of_agent):
            ax.plot(range(1, self.number_of_time_slot + 1), self.mean_average_reward_2d_ndarray[i], '--', color=color_list[i], label=self.agent_name_array[i])
            ax.legend(loc='lower right')

        # Plot the first 300 range of the mean average reward
        fig, ax = plt.subplots(figsize=(16,8), dpi=200)
        for i in range(number_of_agent):
            ax.plot(range(1, 301), self.mean_average_reward_2d_ndarray[i][0:300], '--', color=color_list[i], label=self.agent_name_array[i])
            ax.legend(loc='lower right')

        plt.show()


    def __store_data(self, data_3d_ndarray, mean_2d_ndarray, data_2d_ndarray_to_store):
        # Store the 2d ndarray
        data_3d_ndarray.resize((data_3d_ndarray.shape[0] + 1, data_3d_ndarray.shape[1], data_3d_ndarray.shape[2]), refcheck=False)
        data_3d_ndarray[data_3d_ndarray.shape[0] - 1] = data_2d_ndarray_to_store

        # Calculate the mean among the repetitions and store it
        mean_2d_ndarray.resize((mean_2d_ndarray.shape[0] + 1, mean_2d_ndarray.shape[1]), refcheck=False)
        mean_2d_ndarray[mean_2d_ndarray.shape[0] - 1] = np.mean(data_2d_ndarray_to_store, axis=0)
