from seldonian.utils.RL_utils import *
from seldonian.utils.stats_utils import weighted_sum_gamma

class RL_model: #consist of agent, env
	def __init__(self, agent, env):
		self.agent = agent
		self.env = env

	def get_probs_from_observations_and_actions(self, new_params, observations, actions):
		self.agent.set_new_params(new_params)
		num_probs = len(observations)
		if num_probs != len(actions):
			error(f"different number of observations ({observations}) and actions ({actions})")

		probs = [self.get_prob_this_action(observations[index], actions[index]) for index in range(num_probs)]

		return np.array(probs)

	def get_prob_this_action(self, observation, action):
		return self.agent.get_prob_this_action(observation, action)

	def sample_from_statistic(self,
		statistic_name,theta,data_dict):
		""" Evaluate a provided statistic for each episode 
		in the sample

		:param statistic_name: The name of the statistic to evaluate
		:type statistic_name: str, e.g. 'J_pi_new'

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param data_dict: Contains the dataframe 
		:type data_dict: dict

		:return: The evaluated statistic for each episode in the dataset
		:rtype: numpy ndarray(float)
		"""

		if statistic_name == 'J_pi_new':
			return self.vector_IS_estimate(
				theta,data_dict)
		else:
			raise NotImplementedError(
				f"Statistic: {statistic_name} is not implemented")

	def evaluate_statistic(self,
		statistic_name,theta,data_dict):
		""" Evaluate a provided statistic for the whole sample provided

		:param statistic_name: The name of the statistic to evaluate
		:type statistic_name: str, e.g. 'J_pi_new'

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param data_dict: Contains the dataframe
		:type data_dict: dict

		:return: The evaluated statistic over the whole sample
		:rtype: float
		"""
		if statistic_name == 'J_pi_new':
			return self.sample_IS_estimate(
				theta,data_dict)
		else:
			raise NotImplementedError(
				f"Statistic: {statistic_name} is not implemented")

	def sample_IS_estimate(self,theta,data_dict):
		""" Calculate the unweighted importance sampling estimate
		on all episodes in the dataframe

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param dataset: The object containing data and metadata
		:type dataset: dataset.Dataset object

		:return: The IS estimate calculated over all episodes
		:rtype: float
		"""
		episodes = data_dict['episodes']
		IS_estimate = 0
		for ii, ep in enumerate(episodes):
			pi_news = self.get_probs_from_observations_and_actions(theta, ep.states, ep.actions)
			# print(pi_news,ep.pis)
			pi_ratios = pi_news / ep.pis
			# print(pi_ratios)
			pi_ratio_prod = np.prod(pi_ratios)
			# print(pi_ratio_prod)
			weighted_return = weighted_sum_gamma(ep.rewards, gamma=self.env.gamma)
			# print(weighted_return)
			IS_estimate += pi_ratio_prod * weighted_return

		IS_estimate /= len(episodes)

		return IS_estimate

	def vector_IS_estimate(self, theta, data_dict):
		""" Calculate the unweighted importance sampling estimate
		on each episodes in the dataframe

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param dataframe: Contains the episodes
		:type dataframe: pandas dataframe

		:return: A vector of IS estimates calculated for each episode
		:rtype: numpy ndarray(float)
		"""
		episodes = data_dict['episodes']
		# weighted_reward_sums_by_episode = data_dict['reward_sums_by_episode']
		result = []
		for ii, ep in enumerate(episodes):
			pi_news = self.get_probs_from_observations_and_actions(theta, ep.states, ep.actions)
			# print("pi news:")
			# print(pi_news)
			pi_ratio_prod = np.prod(pi_news / ep.pis)
			# print("pi_ratio_prod:")
			# print(pi_ratio_prod)
			weighted_return = weighted_sum_gamma(ep.rewards, gamma=self.env.gamma)
			# result.append(pi_ratio_prod*weighted_reward_sums_by_episode[ii])
			result.append(pi_ratio_prod * weighted_return)

		return np.array(result)
