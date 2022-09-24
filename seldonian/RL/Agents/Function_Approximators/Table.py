from seldonian.RL.Agents.Function_Approximators.Function_Approximator import *
import autograd.numpy as np

class Table(Function_Approximator):
    def __init__(self, min_state, num_states):
        """ Table holding states, capable of reindexing states
        
        :param min_state: The smallest obs number
        :type min_state: int
        :param num_states: Number of total states
        :type num_states: int
        """
        self.min_state = min_state
        self.num_states = num_states

    def from_environment_state_to_0_indexed_state(self, env_state):
        """
        Convert environment obs to 0 indexed obs in the table

        :param env_state: The environment obs you want to convert
        :type env: int
        :return: 0-indexed obs in the table
        """
        return env_state - self.min_state

class Q_Table(Table):
    def __init__(self, min_state, num_states, num_actions):
        """ A Q table containing transition probabilities 

        :param min_state: The smallest obs number
        :type min_state: int
        :param num_states: Number of total states
        :type num_states: int
        :param num_actions: Number of actions in a given obs
        :type num_actions: int
        """
        super().__init__(min_state, num_states)
        self.weights = np.zeros((num_states, num_actions))
        self.num_actions = num_actions

    def get_action_values_given_state(self, state_number_not_zero_indexed):
        """ Get possible Q-table values given environmental obs

        :param state_number_not_zero_indexed: The environment-specific obs number
        :type state_number_not_zero_indexed: int
        :return: array of possible Q-table values
        """

        zero_indexed_state_number = self.from_environment_state_to_0_indexed_state(state_number_not_zero_indexed)
        return self.get_action_values_given_zero_indexed_state(zero_indexed_state_number)

    def get_action_values_given_zero_indexed_state(self, zero_indexed_state_number):
        """ Get possible Q-table values given 0-indexed obs number in the table

        :param zero_indexed_state_number: The 0-indexed obs number in the table
        :type zero_indexed_state_number: int
        :return: array of possible actions
        """
        return self.weights[zero_indexed_state_number, :]

def construct_Q_Table_From_Env_Description(env_description):
    """ Create a Q table given an environment description 

    :param env_description: an object for accessing attributes
            of the environment
    :type env_description: :py:class:`.Env_Description`
    :return: A Q Table 
    :rtype: :py:class:`.Q_Table`
    """
    return Q_Table(env_description.get_min_state(), env_description.get_num_states(), env_description.get_num_actions())
