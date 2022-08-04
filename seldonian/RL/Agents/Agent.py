from seldonian.RL.Env_Description.Env_Description import *
from seldonian.RL.Agents.Function_Approximators.Table import *
from seldonian.RL.Agents.Function_Approximators.Linear_FA import *
from seldonian.RL.Agents.Bases.Fourier import *

class Agent:
    def choose_action(self, observation):
        raise NotImplementedError()

    def update(self, observation, next_observation, reward, terminated):
        raise NotImplementedError()

    def get_prob_this_action(self, observation, action):
        raise NotImplementedError()

    def set_new_params(self, theta):
        raise NotImplementedError()

    def make_state_action_FA(self, env_desciption, hyperparam_and_setting_dict):
        if type(env_desciption.observation_space) == Discrete_Space and type(env_desciption.action_space) == Discrete_Space:
            return construct_Q_Table_From_Env_Description(env_desciption)
        if type(env_desciption.observation_space) == Continuous_Space and type(env_desciption.action_space) == Discrete_Space:
            return self.construct_basis_and_linear_FA(env_desciption, hyperparam_and_setting_dict)
        else:
            error(f"unhandled state type {type(env_desciption.observation_space)} and action type {type(env_desciption.action_space)} for make_state_action_FA()")

    def construct_basis_and_linear_FA(self, env_description, hyperparam_and_setting_dict):
        basis_type = hyperparam_and_setting_dict["basis"]
        if basis_type == "Fourier":
            basis = Fourier(hyperparam_and_setting_dict, env_description)
        else:
            error("unknown basis type ", basis_type)
        return Linear_state_action_value_FA(basis, env_description)

    def get_params(self):
        raise NotImplementedError()
