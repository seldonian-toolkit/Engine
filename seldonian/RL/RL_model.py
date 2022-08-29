import autograd.numpy as np

class RL_model: #consist of agent, env
	def __init__(self, policy, env_kwargs):
		self.policy = policy
		self.env_kwargs = env_kwargs
		if 'gamma' not in self.env_kwargs:
			self.env_kwargs['gamma'] = 1.0

	def get_probs_from_observations_and_actions(self, new_params, observations, actions):
		self.policy.set_new_params(new_params)
		num_probs = len(observations)
		if num_probs != len(actions):
			error(f"different number of observations ({observations}) and actions ({actions})")

		probs = [self.get_prob_this_action(observations[index], actions[index]) for index in range(num_probs)]
		# probs = list(map(self.policy.get_probs_given_states_actions,
  #                   observations,
  #                   actions))
		# self.policy._denom.cache_clear()
		# self.policy._arg.cache_clear()
		return np.array(probs)

	def get_prob_this_action(self, observation, action):
		return self.policy.get_prob_this_action(observation, action)
