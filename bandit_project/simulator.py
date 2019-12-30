import numpy as np
import matplotlib.pyplot as plt
from util.plot_utils import plot_vertical_bar_chart
import pickle


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
        self.mean_aggregated_reward_in_total_array = np.array([])
        self.payload_list = []
        self.agent_name_array = np.array([])

        self.color_list = ['#2c3e50', '#1e8bc3', '#19b5fe', '#cf000f', '#db0a5b', '#f1a9a0', '#00e640', '#03a678', '#00b16a']


    def run_agent(self, agent):
        reward_2d_ndarray = np.zeros((self.number_of_repetition, self.number_of_time_slot), dtype=float)
        aggregated_reward_2d_ndarray = np.zeros((self.number_of_repetition, self.number_of_time_slot), dtype=float)
        average_reward_2d_ndarray = np.zeros((self.number_of_repetition, self.number_of_time_slot), dtype=float)
        payload = []

        for i in range(self.number_of_repetition):
            # Reset the agent
            agent.reset()

            # Initialization
            reward_array = np.zeros(self.number_of_time_slot, dtype=float)
            aggregated_reward_array = np.zeros(self.number_of_time_slot, dtype=float)
            average_reward_array = np.zeros(self.number_of_time_slot, dtype=float)
            round_payload = []

            # Let the agent pull arms repeatedly
            for t in range(1, self.number_of_time_slot + 1):
                I, reward, round_payload_item = agent.pull_arm()

                reward_array[t-1] = reward
                aggregated_reward_array[t-1] = np.sum(reward_array[0:t])
                average_reward_array[t-1] = np.average(reward_array[0:t])
                round_payload.append(round_payload_item)
            
            # Store the reward array for this iteration in the 2d ndarray for reward
            reward_2d_ndarray[i] = reward_array

            # Store the aggregated reward array for this iteration in the 2d ndarray for aggregated reward
            aggregated_reward_2d_ndarray[i] = aggregated_reward_array

            # Store the average reward array for this iteration in the 2d ndarray for average reward
            average_reward_2d_ndarray[i] = average_reward_array

            # Store the round payload
            payload.append(round_payload)
        
        # Store the data for reward
        self.__store_data(self.reward_3d_ndarray, self.mean_reward_2d_ndarray, reward_2d_ndarray)

        # Store the data for aggregated rewrad
        self.__store_data(self.aggregated_reward_3d_ndarray, self.mean_aggregated_reward_2d_ndarray, aggregated_reward_2d_ndarray)
        
        # Store the data for average reward
        self.__store_data(self.average_reward_3d_ndarray, self.mean_average_reward_2d_ndarray, average_reward_2d_ndarray)

        # Store the name of the agent
        self.agent_name_array = np.append(self.agent_name_array, agent.name())

        # Store the mean aggregated reward in total
        self.mean_aggregated_reward_in_total_array = np.append(self.mean_aggregated_reward_in_total_array, np.mean(np.sum(reward_2d_ndarray, axis=1)))

        # Store the payload for this agent
        self.payload_list.append(payload)

        # Print information of executation
        print('Agent \'{}\' completed.'.format(agent.name()))
        print('    Mean aggregated reward in total: {}'.format(self.mean_aggregated_reward_in_total_array[-1]))

        return


    def plot(self, indices=None):
        # Plot the mean aggregated reward
        self.plot_mean_aggregated_reward(indices=indices)

        # Plot the mean average reward
        self.plot_mean_average_reward(indices=indices)

        # Plot the first 300 range of the mean average reward
        self.plot_first_300_range_mean_average_reward(indices=indices)

        # Plot the mean aggregated reward in total array
        self.plot_mean_aggregated_reward_in_total(oracle_value=None, indices=indices)

        plt.show()

        return


    def __store_data(self, data_3d_ndarray, mean_2d_ndarray, data_2d_ndarray_to_store):
        # Store the 2d ndarray
        data_3d_ndarray.resize((data_3d_ndarray.shape[0] + 1, data_3d_ndarray.shape[1], data_3d_ndarray.shape[2]), refcheck=False)
        data_3d_ndarray[data_3d_ndarray.shape[0] - 1] = data_2d_ndarray_to_store

        # Calculate the mean among the repetitions and store it
        mean_2d_ndarray.resize((mean_2d_ndarray.shape[0] + 1, mean_2d_ndarray.shape[1]), refcheck=False)
        mean_2d_ndarray[mean_2d_ndarray.shape[0] - 1] = np.mean(data_2d_ndarray_to_store, axis=0)

        return


    def plot_mean_aggregated_reward(self, indices=None):
        number_of_agent = self.reward_3d_ndarray.shape[0]

        fig, ax = plt.subplots(figsize=(16,8), dpi=200)
        index_range = range(number_of_agent) if indices is None else indices
        for i in index_range:
            ax.plot(range(1, self.number_of_time_slot + 1), self.mean_aggregated_reward_2d_ndarray[i], '--', color=self.color_list[i], label=self.agent_name_array[i])
            ax.legend(loc='lower right')
            ax.set_title('Mean aggregated reward v.s. time slot $t$')
            ax.set_xlabel('Time slot $t$')
            ax.set_ylabel('Mean aggregated reward')

        plt.show()

        return


    def plot_mean_average_reward(self, indices=None):
        number_of_agent = self.reward_3d_ndarray.shape[0]

        fig, ax = plt.subplots(figsize=(16,8), dpi=200)
        index_range = range(number_of_agent) if indices is None else indices
        for i in index_range:
            ax.plot(range(1, self.number_of_time_slot + 1), self.mean_average_reward_2d_ndarray[i], '--', color=self.color_list[i], label=self.agent_name_array[i])
            ax.legend(loc='lower right')
            ax.set_title('Mean average reward v.s. time slot $t$')
            ax.set_xlabel('Time slot $t$')
            ax.set_ylabel('Mean average reward')

        plt.show()

        return


    def plot_first_300_range_mean_average_reward(self, indices=None):
        number_of_agent = self.reward_3d_ndarray.shape[0]

        fig, ax = plt.subplots(figsize=(16,8), dpi=200)
        index_range = range(number_of_agent) if indices is None else indices
        for i in index_range:
            ax.plot(range(1, 301), self.mean_average_reward_2d_ndarray[i][0:300], '--', color=self.color_list[i], label=self.agent_name_array[i])
            ax.legend(loc='lower right')
            ax.set_title('Mean average reward v.s. time slot $t \in [1, 300]$')
            ax.set_xlabel('Time slot $t$')
            ax.set_ylabel('Mean average reward')

        plt.show()

        return


    def plot_mean_aggregated_reward_in_total(self, oracle_value=None, indices=None):
        selected_agent_name = None
        selected_values = None
        if indices is not None:
            index_range = indices
            selected_agent_name = []
            selected_values = []
            for i in index_range:
                selected_agent_name.append(self.agent_name_array[i])
                selected_values.append(self.mean_aggregated_reward_in_total_array[i])
        else:
            selected_agent_name = self.agent_name_array
            selected_values = self.mean_aggregated_reward_in_total_array
        
        plot_vertical_bar_chart(selected_agent_name, selected_values, title='Mean aggregated reward in total v.s. different agents', xlabel='Different agents', ylabel='Mean aggregated reward', rotation=90, horizontal_line_value=oracle_value)

        return


    def dump(self, filename=None):
        if filename is None:
            filename = 'pickled_simulator_with_id_{}.p'.format(id(self))
        
        pickle.dump(self, open(filename, "wb"))

        print('The Simulator object has been pickled to file \'{}\'.'.format(filename))

        return
