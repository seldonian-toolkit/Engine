from seldonian.RL.Agents.Agent import *
from seldonian.RL.Agents.Function_Approximators.Table import *
from seldonian.RL.Agents.Policies.Softmax import *


class Parameterized_non_learning_softmax_agent(Agent):
    def __init__(self, env_description, hyperparam_and_setting_dict):
        """
        RL agent that takes actions using parametrized softmax function:
        :math:`\pi(s,a) = \\frac{e^{p(s,a)}}{\sum_{a'}{e^{p(s,a')}}}`

        :param env_description: an object for accessing attributes
                of the environment
        :type env_description: :py:class:`.Env_Description`
        :param hyperparam_and_setting_dict: Contains additional
                info about the environment and data generation
        :ivar FA: Function approximator
        :vartype FA: :py:class:`.Q_Table`
        :ivar softmax: The policy
        :vartype softmax: :py:class:`.Softmax`
        """

        self.softmax = Softmax(hyperparam_and_setting_dict, env_description)
        self.env_description = env_description

    def get_action_values(self, obs):
        """Get all possible actions from this state using the FA

        :param obs: The current observation of the agent,
                type depends on environment.
        """
        return self.softmax.get_action_values_given_state(obs)

    def choose_action(self, obs):
        """Select an action given a observation

        :param obs: The current observation of the agent,
                type depends on environment
        :return: array of actions
        """
        return self.softmax.choose_action(obs)

    def update(self, observation, next_observation, reward, terminated):
        """
        Updates agent's parameters according to the learning rule.
        Not implemented for this agent.

        :param observation: The current observation of the agent,
                type depends on environment.
        :param next_observation: The observation of the agent after
                an action is taken
        :param reward: The reward for taking the action
        :param terminated: Whether next_observation is the
                terminal observation
        :type terminated: bool
        """
        pass

    def get_prob_this_action(self, observation, action):
        """Get the probability of a selected action in a given obs

        :param observation: The current obs of the agent, type depends on environment.
        :param action: The action selected, type depends on environment
        :return: probability of action
        :rtype: float
        """
        return self.softmax.get_prob_this_action(observation, action)

    def set_new_params(self, new_params):
        """Set the parameters of the agent

        :param new_params: array of weights
        """
        self.softmax.set_new_params(new_params)

    def get_params(self):
        """Get the current parameters (weights) of the agent

        :return: array of weights
        """
        return self.softmax.get_params()

    def get_policy(self):
        return self.softmax
