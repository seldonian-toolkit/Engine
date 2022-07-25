from seldonian.RL.Env_Description.Spaces import *
from utils import *


class Env_Description:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def get_num_states(self):
        check_space_type(self.observation_space, Discrete_Space)
        return self.observation_space.get_num_values()

    def get_num_actions(self):
        check_space_type(self.observation_space, Discrete_Space)
        return self.action_space.get_num_values()

    def get_min_action(self):
        check_space_type(self.action_space, Discrete_Space)
        return self.action_space.min

    def get_min_state(self):
        check_space_type(self.action_space, Discrete_Space)
        return self.observation_space.min


def check_space_type(space, desired_type):
    if type(space) != desired_type:
        error(f"need space of type {desired_type}, but got space of type {type(space)}")
