from seldonian.RL.Agents.Agent import *
from seldonian.RL.Agents.Function_Approximators.Table import *
from seldonian.RL.Agents.Policies.Softmax import *

class Parameterized_non_learning_softmax_agent(Agent):
    def __init__(self, env_description):
        num_actions = env_description.get_num_actions()
        self.FA = self.make_state_action_FA(env_description)
        self.softmax = Softmax(env_description.get_min_action(), num_actions)
        self.env_description = env_description

    def get_action_values(self, state):
        return self.FA.get_action_values_given_state(state)

    def choose_action(self, state):
        action_values = self.get_action_values(state)
        return self.softmax.choose_action(action_values)

    def update(self, state, next_state, reward, terminated):
        pass

    def get_prob_this_action(self, state, action):
        action_values = self.get_action_values(state)
        action_probs = self.softmax.get_action_probs_from_action_values(action_values)
        this_action = self.softmax.from_environment_action_to_0_indexed_action(action)
        return action_probs[this_action]

    def set_new_params(self, theta):
        self.FA.set_new_params(theta)