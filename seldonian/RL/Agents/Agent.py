from seldonian.RL.Env_Description.Env_Description import *
from seldonian.RL.Agents.Function_Approximators.Table import *

class Agent:
    def choose_action(self, observation):
        raise NotImplementedError()

    def update(self, observation, next_observation, reward, terminated):
        raise NotImplementedError()

    def get_prob_this_action(self, observation, action):
        raise NotImplementedError()

    def set_new_params(self, theta):
        raise NotImplementedError()

    def make_state_action_FA(self, env_desciption):
        if type(env_desciption.observation_space) == Discrete_Space and type(env_desciption.action_space) == Discrete_Space:
            return construct_Q_Table_From_Env_Description(env_desciption)
        else:
            error(f"unhandled state type {type(env_desciption.observation_space)} and action type {type(env_desciption.action_space)} for make_state_action_FA()")
