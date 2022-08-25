from seldonian.RL.Agents.Function_Approximators.Table import *
from seldonian.RL.Agents.Function_Approximators.Linear_FA import *
from seldonian.RL.Agents.Bases.Fourier import *
from seldonian.RL.Env_Description.Env_Description import *

class Policy:
    pass

class Discrete_Action_Policy(Policy):
    def __init__(self, hyperparam_and_setting_dict, env_description):
        self.min_action = env_description.get_min_action() #e.g., if environment actions are {-1, 0, 1}, then this is -1
        self.num_actions = env_description.get_num_actions()
        self.FA = self.make_state_action_FA(env_description, hyperparam_and_setting_dict)

    def from_0_indexed_action_to_environment_action(self, action_0_indexed):
        return action_0_indexed + self.min_action

    def from_environment_action_to_0_indexed_action(self, env_action):
        return env_action - self.min_action

    def make_state_action_FA(self, env_description, hyperparam_and_setting_dict):
        """ Create a function approximator from an environment description and
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
        if type(env_description.observation_space) == Discrete_Space and type(
                env_description.action_space) == Discrete_Space:
            return construct_Q_Table_From_Env_Description(env_description)
        if type(env_description.observation_space) == Continuous_Space and type(
                env_description.action_space) == Discrete_Space:
            return self.construct_basis_and_linear_FA(env_description, hyperparam_and_setting_dict)
        else:
            error(f"unhandled state type {type(env_description.observation_space)} and action type {type(env_description.action_space)} for make_state_action_FA()")

    def construct_basis_and_linear_FA(self, env_description, hyperparam_and_setting_dict):
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
        else:
            error("unknown basis type ", basis_type)
        return Linear_state_action_value_FA(basis, env_description)