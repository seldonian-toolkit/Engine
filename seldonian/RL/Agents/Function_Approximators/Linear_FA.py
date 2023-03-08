from seldonian.RL.Agents.Function_Approximators.Function_Approximator import *


class Linear_FA(Function_Approximator):
    def __init__(self, basis, env_description):
        """Base class for linear function approximators.

        :param basis: The basis to use for encoding features
        :param env_description: an object for accessing attributes
            of the environment
        :type env_description: :py:class:`.Env_Description`
        """
        self.basis = basis
        self.weights = np.zeros((basis.num_features, env_description.get_num_actions()))


class Linear_state_action_value_FA(Linear_FA):
    def __init__(self, basis, env_description):
        """A linear state action value function approximator.

        :param basis: The basis to use for encoding features
        :param env_description: an object for accessing attributes
            of the environment
        :type env_description: :py:class:`.Env_Description`
        """
        super().__init__(basis, env_description)

    def get_action_values_given_state(self, state):
        return self.get_action_values_given_features(self.get_features(state))

    def get_action_values_given_features(self, features):
        return np.dot(features, self.weights)

    def get_features(self, state):
        return self.basis.get_features(state)
