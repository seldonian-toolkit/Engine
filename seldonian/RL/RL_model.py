import autograd.numpy as np
from seldonian.models.models import SeldonianModel


class RL_model(SeldonianModel):  # consist of agent, env
    def __init__(self, policy, env_kwargs):
        """Base class for all RL models.

        :param policy: A policy parameterization
        :type policy: :py:class:`.Policy`

        :param env_kwargs: Kwargs pertaining to environment
                such as gamma, the discount factor
        :type env_kwargs: dict
        """

        self.policy = policy
        self.env_kwargs = env_kwargs
        if "gamma" not in self.env_kwargs:
            self.env_kwargs["gamma"] = 1.0

    def get_probs_from_observations_and_actions(
        self, new_params, observations, actions, action_probs,
    ):
        """Get action probablities given a list of observations and actions
        taken given those observations

        :param new_params: Parameter weights to use
        :param observations: Array of observations
        :param actions: Array of actions
        :param action_probs: Array of action probabilities from the behavior policy

        :return: Array of probabilities
        """
        self.policy.set_new_params(new_params)
        num_probs = len(observations)
        if num_probs != len(actions):
            error(
                f"different number of observations ({observations}) and actions ({actions})"
            )

        probs = list(map(self.policy.get_prob_this_action, observations, actions, action_probs))
        # If the policy uses a cache, make sure to clear it
        # This is necessary because cache is only correct
        # for a given set of param weights
        try:
            self.policy._denom.cache_clear()
            self.policy._arg.cache_clear()
        except:
            pass

        return np.array(probs)

    def get_prob_this_action(self, observation, action, action_prob):
        """Get action probability given a single observation, action
        pair

        :param observation: Observation on a single timestep
        :param action: Action on a single timestep
        :param action_prob: Action probability on a single timestep

        :return: Probability of taking this action given observation
        :rtype: float
        """

        return self.policy.get_prob_this_action(observation, action, action_prob)
