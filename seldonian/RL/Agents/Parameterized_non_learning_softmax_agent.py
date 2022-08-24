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
		num_actions = env_description.get_num_actions()
		self.FA = self.make_state_action_FA(env_description, hyperparam_and_setting_dict)
		self.softmax = Softmax(env_description.get_min_action(), num_actions)
		self.env_description = env_description

	def get_action_values(self, obs):
		""" Get all possible actions from this state using the FA

		:param obs: The current observation of the agent, type depends on environment.
		"""
		return self.FA.get_action_values_given_state(obs)

	def choose_action(self, obs):
		""" Select an action given a observation

		:param obs: The current observation of the agent, type depends on environment.

		:return: array of actions
		"""
		action_values = self.get_action_values(obs)
		return self.softmax.choose_action(action_values)

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
		""" Get the probability of a selected action in a given obs

		:param observation: The current obs of the agent, type depends on environment.

		:param action: The action selected, type depends on environment

		:return: probability of action
		:rtype: float
		"""
		action_values = self.get_action_values(observation)
		action_probs = self.softmax.get_action_probs_from_action_values(action_values)
		this_action = self.softmax.from_environment_action_to_0_indexed_action(action)
		return action_probs[this_action]

	def set_new_params(self, new_params):
		""" Set the parameters of the agent

		:param new_params: array of weights
		"""
		self.FA.set_new_params(new_params)

	def get_params(self):
		""" Get the current parameters (weights) of the agent

		:return: array of weights
		"""
		return self.FA.weights