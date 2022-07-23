from seldonian.RL.Agents.Function_Approximators.Function_Approximator import *
import numpy as np

class Table(Function_Approximator):
    def __init__(self, min_state, num_states):
        self.min_state = min_state
        self.num_states = num_states

    def from_environment_state_to_0_indexed_state(self, env_state):
        return env_state - self.min_state

class Q_Table(Table):
    def __init__(self, min_state, num_states, num_actions):
        super().__init__(min_state, num_states)
        self.weights = np.zeros((num_actions, num_states))
        self.num_actions = num_actions

    def get_action_values_given_state(self, state_number_not_zero_indexed):
        zero_indexed_state_number = self.from_environment_state_to_0_indexed_state(state_number_not_zero_indexed)
        return self.get_action_values_given_zero_indexed_state(zero_indexed_state_number)

    def get_action_values_given_zero_indexed_state(self, zero_indexed_state_number):
        return self.weights[:, zero_indexed_state_number]

def construct_Q_Table_From_Env_Description(env_description):
    return Q_Table(env_description.get_min_state(), env_description.get_num_states(), env_description.get_num_actions())
