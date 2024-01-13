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
        """Get action probablities under policy with new parameters.
        Just a wrapper to call policy method of same name.

        :param new_params: New policy parameter weights to set
        :param observations: Array of observations
        :param actions: Array of actions
        :param action_probs: Array of action probabilities from the behavior policy

        :return: Array of action probabilities under the new policy
        """
        self.policy.set_new_params(new_params)
        num_probs = len(observations)
        if num_probs != len(actions):
            error(
                f"different number of observations ({observations}) and actions ({actions})"
            )

        probs = self.policy.get_probs_from_observations_and_actions(
            observations, actions, action_probs
        )

        return np.array(probs)
