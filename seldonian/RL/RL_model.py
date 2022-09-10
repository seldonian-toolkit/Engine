import autograd.numpy as np
from seldonian.models.models import SeldonianModel

class RL_model(SeldonianModel): #consist of agent, env
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

		probs = list(map(self.policy.get_prob_this_action,
                    observations,
                    actions))
		try:
			self.policy._denom.cache_clear()
			self.policy._arg.cache_clear()
		except:
			pass

		return np.array(probs)

	def get_prob_this_action(self, observation, action):
		return self.policy.get_prob_this_action(observation, action)
