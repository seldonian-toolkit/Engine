from seldonian.utils.RL_utils import *
from seldonian.utils.stats_utils import weighted_sum_gamma

class RL_model: #consist of agent, env
	def __init__(self, policy, env):
		self.policy = policy
		self.env = env

	def get_probs_from_observations_and_actions(self, new_params, observations, actions):
		self.policy.set_new_params(new_params)
		num_probs = len(observations)
		if num_probs != len(actions):
			error(f"different number of observations ({observations}) and actions ({actions})")

		probs = [self.get_prob_this_action(observations[index], actions[index]) for index in range(num_probs)]

		return np.array(probs)

	def get_prob_this_action(self, observation, action):
		return self.policy.get_prob_this_action(observation, action)
