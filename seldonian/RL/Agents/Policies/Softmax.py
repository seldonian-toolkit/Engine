from seldonian.RL.Agents.Policies.Policy import *
import autograd.numpy as np
from seldonian.utils.RL_utils import *
from functools import lru_cache

class Softmax(Discrete_Action_Policy):
    def __init__(self, hyperparam_and_setting_dict, env_description):
        super().__init__(hyperparam_and_setting_dict, env_description)

    def choose_action(self, obs):
        """ Select an action given a observation

        :param obs: The current observation of the agent, type depends on environment.

        :return: array of actions
        """
        action_values = self.get_action_values_given_state(obs)
        return self.choose_action_from_action_values(action_values)

    def choose_action_from_action_values(self, action_values):
        if len(action_values) != self.num_actions:
            error(f"should have {self.num_actions} actions, but got {len(action_values)} action values")

        action_probs = self.get_action_probs_from_action_values(action_values)

        roulette_wheel_start = 0.0
        stop_value = np.random.rand()
        for action_num_zero_indexed in range(self.num_actions):
            roulette_wheel_start += action_probs[action_num_zero_indexed]
            if roulette_wheel_start >= stop_value:
                return self.from_0_indexed_action_to_environment_action(action_num_zero_indexed)

        print(stop_value)
        print(roulette_wheel_start)
        print(action_probs)
        error("reached the end of SoftMax.choose_action(), this should never happen")

    def get_action_probs_from_action_values(self, action_values):
        e_to_the_something_terms = self.get_e_to_the_something_terms(action_values)
        denom = sum(e_to_the_something_terms)
        return e_to_the_something_terms / denom

    def get_e_to_the_something_terms(self, action_values):
        max_value = np.max(action_values)
        e_to_the_something_terms = np.exp(action_values - max_value) #subtract max for numerical stability
        return e_to_the_something_terms

    def get_prob_this_action(self, observation, action):
        """ Get the probability of a selected action in a given obs

        :param observation: The current obs of the agent, type depends on environment.

        :param action: The action selected, type depends on environment

        :return: probability of action
        :rtype: float
        """
        action_values = self.get_action_values_given_state(observation)
        action_probs = self.get_action_probs_from_action_values(action_values)
        this_action = self.from_environment_action_to_0_indexed_action(action)
        return action_probs[this_action]



class DiscreteSoftmax(Softmax):
    def __init__(self, hyperparam_and_setting_dict, env_description):
        """ Softmax where observations and actions are discrete.
        Uses cache for lookups to Q Table """
        super().__init__(hyperparam_and_setting_dict, env_description)

    @lru_cache
    def _denom(self,observation):
        """ Helper function to accelerate action probability calculation 

        :param observation: An observation of the environment
        :type observation: int
        """
        return np.sum(np.exp(self.FA.weights[observation]))

    @lru_cache
    def _arg(self,observation,action):
        """ Helper function to accelerate action probability calculation 

        :param observation: A observation of the environment
        :type observation: int

        :param action: A possible action at the given observation
        :type action: int
        """
        return self.FA.weights[observation][action]

    def get_prob_this_action(self,observation,action):
        return np.exp(self._arg(observation,action))/self._denom(observation)


