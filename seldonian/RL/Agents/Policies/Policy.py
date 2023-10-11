from seldonian.RL.Agents.Function_Approximators.Table import *
from seldonian.RL.Agents.Function_Approximators.Linear_FA import *
from seldonian.RL.Agents.Bases.Fourier import *
from seldonian.RL.Env_Description.Env_Description import *


class Policy(object):
    def __init__(self):
        """Base class for policies. Contains methods, some of which
        must be overridden in any policy implementation"""
        pass

    def choose_action(self, obs):
        """Defines how to select an action given an observation, obs"""
        raise NotImplementedError()

    def set_new_params(self, new_params):
        """Update policy parameters"""
        raise NotImplementedError()

    def get_params(self):
        """Get current policy parameters"""
        raise NotImplementedError()

    def get_prob_this_action(self, obs, action):
        """Get probability of taking an action given an observation.
        Does not necessarily need to be overridden, but is often called from 
        self.get_probs_from_observations_and_actions()"""
        raise NotImplementedError()

    def get_probs_from_observations_and_actions(self,observations,actions,behavior_action_probs):
        """Get probabilities for each observation and action in the input arrays"""
        raise NotImplementedError()

class Discrete_Action_Policy(Policy):
    def __init__(self, hyperparam_and_setting_dict, env_description):
        """General policy class where actions are discrete. Converts 
        input actions into 0-indexed actions.

        :param hyperparameter_and_setting_dict: Specifies the
            environment, agent, number of episodes per trial,
            and number of trials
        :param env_description: an object for accessing attributes
            of the environment
        :type env_description: :py:class:`.Env_Description`
        """
        self.min_action = (
            env_description.get_min_action()
        )  # e.g., if environment actions are {-1, 0, 1}, then this is -1
        self.num_actions = env_description.get_num_actions()
        self.FA = self.make_state_action_FA(
            env_description, hyperparam_and_setting_dict
        )

    def from_0_indexed_action_to_environment_action(self, action_0_indexed):
        """Convert 0-indexed action to env-specific action"""
        return action_0_indexed + self.min_action

    def from_environment_action_to_0_indexed_action(self, env_action):
        """Convert env-specific action to 0 indexed action"""
        return env_action - self.min_action

    def make_state_action_FA(self, env_description, hyperparam_and_setting_dict):
        """Create a function approximator from an environment description and
        dictionary specification

        :param env_description: an object for accessing attributes
            of the environment
        :type env_description: :py:class:`.Env_Description`
        :param hyperparameter_and_setting_dict: Specifies the
            environment, agent, number of episodes per trial,
            and number of trials

        :return: function approximator, type depends on whether observation
            space is discrete or continous
        """
        if (
            type(env_description.observation_space) == Discrete_Space
            and type(env_description.action_space) == Discrete_Space
        ):
            return construct_Q_Table_From_Env_Description(env_description)
        if (
            type(env_description.observation_space) == Continuous_Space
            and type(env_description.action_space) == Discrete_Space
        ):
            return self.construct_basis_and_linear_FA(
                env_description, hyperparam_and_setting_dict
            )
        else: # pragma: no cover
            error(
                f"unhandled state type {type(env_description.observation_space)} and action type {type(env_description.action_space)} for make_state_action_FA()"
            )

    def construct_basis_and_linear_FA(
        self, env_description, hyperparam_and_setting_dict
    ):
        """Create a basis and linear function approximator
        from an environment description and dictionary specification

        :param env_description: an object for accessing attributes
            of the environment
        :type env_description: :py:class:`.Env_Description`
        :param hyperparameter_and_setting_dict: Specifies the
            environment, agent, number of episodes per trial,
            and number of trials

        """
        basis_type = hyperparam_and_setting_dict["basis"]
        if basis_type == "Fourier":
            basis = Fourier(hyperparam_and_setting_dict, env_description)
        else: # pragma: no cover
            error("unknown basis type ", basis_type)
        return Linear_state_action_value_FA(basis, env_description)

    def get_action_values_given_state(self, obs):
        """Get all parameter weights possible in a given observation"""
        return self.FA.get_action_values_given_state(obs)

    def set_new_params(self, new_params):
        """Set the parameters of the agent

        :param new_params: array of weights
        """
        self.FA.set_new_params(new_params)

    def get_params(self):
        """Get the current parameters (weights) of the agent

        :return: array of weights
        """
        return self.FA.weights
