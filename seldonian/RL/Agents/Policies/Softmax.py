from seldonian.RL.Agents.Policies.Policy import *
import autograd.numpy as np
from seldonian.utils.RL_utils import *
from functools import lru_cache


class Softmax(Discrete_Action_Policy):
    def __init__(self, hyperparam_and_setting_dict, env_description):
        """General softmax policy

        :param hyperparameter_and_setting_dict: Specifies the
            environment, agent, number of episodes per trial,
            and number of trials
        :param env_description: an object for accessing attributes
            of the environment
        :type env_description: :py:class:`.Env_Description`
        """
        super().__init__(hyperparam_and_setting_dict, env_description)

    def choose_action(self, obs):
        """Select an action given an observation

        :param obs: An observation of the environment

        :return: array of actions
        """
        action_values = self.get_action_values_given_state(obs)
        return self.choose_action_from_action_values(action_values)

    def choose_action_from_action_values(self, action_values):
        """Select an action given a list of action values

        :param action_values: List of action values (param weights)
        """
        if len(action_values) != self.num_actions:
            error(
                f"should have {self.num_actions} actions, but got {len(action_values)} action values"
            )

        action_probs = self.get_action_probs_from_action_values(action_values)

        roulette_wheel_start = 0.0
        stop_value = np.random.rand()
        for action_num_zero_indexed in range(self.num_actions):
            roulette_wheel_start += action_probs[action_num_zero_indexed]
            if roulette_wheel_start >= stop_value:
                return self.from_0_indexed_action_to_environment_action(
                    action_num_zero_indexed
                )

        print(stop_value, roulette_wheel_start, action_probs)  # pragma: no cover
        error(
            "reached the end of SoftMax.choose_action(), this should never happen"
        )  # pragma: no cover

    def get_action_probs_from_action_values(self, action_values):
        """Get action probabilities given a list of action values

        :param action_values: List of action values (param weights)

        :return: array of action probabilites
        """
        e_to_the_something_terms = self.get_e_to_the_something_terms(action_values)
        denom = sum(e_to_the_something_terms)
        return e_to_the_something_terms / denom

    def get_e_to_the_something_terms(self, action_values):
        """Exponentiate list of action values

        :param action_values: List of action values (param weights)

        :return: array of exponentiated action values
        """
        max_value = np.max(action_values)
        e_to_the_something_terms = np.exp(
            action_values - max_value
        )  # subtract max for numerical stability
        return e_to_the_something_terms

    def get_prob_this_action(self, observation, action):
        """Get the probability of a selected action in a given obsertavtion

        :param observation: The current observation of the environment
        :param action: The selected action

        :return: probability of action
        :rtype: float
        """
        action_values = self.get_action_values_given_state(observation)
        action_probs = self.get_action_probs_from_action_values(action_values)
        this_action = self.from_environment_action_to_0_indexed_action(action)
        return action_probs[this_action]

    def get_probs_from_observations_and_actions(
        self, observations, actions, behavior_action_probs
    ):
        """Get the action probabilities of a selected actions and observations under
        the new policy

        :param observations: array of observations of the environment
        :param actions: array of selected actions
        :param behavior_action_probs: The probability of the selected actions under the behavior policy

        :return: array action probabilities of the observation,action pairs under the new policy
        :rtype: numpy.ndarray(float)
        """
        action_probs = np.array(
            list(map(self.get_prob_this_action, observations, actions))
        )
        return action_probs


class DiscreteSoftmax(Softmax):
    def __init__(self, hyperparam_and_setting_dict, env_description):
        """Softmax where both observations and actions are discrete.
        Faster than just using Softmax class because
        a cache is used for lookups to Q Table"""
        super().__init__(hyperparam_and_setting_dict, env_description)

    @lru_cache
    def _denom(self, observation):
        """Helper function to accelerate action probability calculation

        :param observation: An observation of the environment
        :type observation: int
        """
        return np.sum(np.exp(self.FA.weights[observation]))

    @lru_cache
    def _arg(self, observation, action):
        """Helper function to accelerate action probability calculation

        :param observation: A observation of the environment
        :type observation: int
        :param action: A possible action at the given observation
        :type action: int
        """
        return self.FA.weights[observation][action]

    def get_prob_this_action(self, observation, action):
        """Get the probability of a selected action in a given obsertavtion

        :param observation: The current obseravation of the environment
        :param action: The selected action
        :param action_prob: The probability of the selected action

        :return: probability of action
        :rtype: float
        """
        return np.exp(self._arg(observation, action)) / self._denom(observation)

    def get_probs_from_observations_and_actions(
        self, observations, actions, behavior_action_probs
    ):
        """Get the action probabilities of selected actions and observations under
        the new policy.

        :param observations: array of observations of the environment
        :param actions: array of selected actions
        :param behavior_action_probs: The probability of the selected actions under the behavior policy

        :return: action probabilities of the observation,action pairs under the new policy
        :rtype: numpy.ndarray(float)
        """
        action_probs = np.array(
            list(map(self.get_prob_this_action, observations, actions))
        )

        # Clear the cache
        # This is necessary because cache is only correct
        # for a given set of param weights
        self._denom.cache_clear()
        self._arg.cache_clear()
        return action_probs


class MixedSoftmax(Softmax):
    def __init__(self, hyperparam_and_setting_dict, env_description, alpha=0.5):
        """Softmax mixed policy, as in https://people.cs.umass.edu/~pthomas/papers/Thomas2015b.pdf
        see equation for mu in last paragraph before Section 2.1. Stationarity

        :param hyperparameter_and_setting_dict: Specifies the
            environment, agent, number of episodes per trial,
            and number of trials

        :param env_description: an object for accessing attributes
            of the environment
        :type env_description: :py:class:`.Env_Description`

        :param alpha: The mixing hyperparameter. 100%alpha is how far
            from behavior policy the mixed policy can be.
        """
        super().__init__(hyperparam_and_setting_dict, env_description)
        self.alpha = alpha

    def get_prob_this_action(self, observation, action, pi_b):
        """Get the probability of a selected action under the mixed policy
        given observation, action and action probability of the behavior policy

        :param observation: The current obseravation of the environment
        :param action: The selected action
        :param pi_b: The action probability of the behavior policy

        :return: probability of action under the mixed policy
        :rtype: float
        """
        action_values = self.get_action_values_given_state(observation)
        action_probs = self.get_action_probs_from_action_values(action_values)
        this_action = self.from_environment_action_to_0_indexed_action(action)
        pi_new = action_probs[this_action]
        pi_mixed = self.alpha * pi_new + (1 - self.alpha) * pi_b
        return pi_mixed
