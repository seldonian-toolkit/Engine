from seldonian.RL.Env_Description.Spaces import *
from seldonian.utils.RL_utils import *


class Env_Description:
    def __init__(self, observation_space, action_space):
        """ Describes an environment's observation and action space
        and provides convenience methods for accessing the
        environment's attributes 

        :param observation_space: Discrete or continuous space
            describing observations made in the environment
        :param action_space: Discrete or continuous space
            describing possible actions taken in the environment
        """
        self.observation_space = observation_space
        self.action_space = action_space

    def get_num_states(self):
        """ Get number of states 
        (also called observations here)
        """
        check_space_type(self.observation_space, Discrete_Space)
        return self.observation_space.get_num_values()

    def get_num_actions(self):
        """ Get number of actions
        """
        check_space_type(self.action_space, Discrete_Space)
        return self.action_space.get_num_values()

    def get_min_action(self):
        """ Get first action in the action space 
        """
        check_space_type(self.action_space, Discrete_Space)
        return self.action_space.min

    def get_min_state(self):
        """ Get first obs in the observation space
        """
        check_space_type(self.action_space, Discrete_Space)
        return self.observation_space.min

    def get_num_observation_dims(self):
        """ Get the number of dimensions in the observation space
        """
        return self.observation_space.get_num_dims()


def check_space_type(space, desired_type):
    """ Validator to ensure space types are equivalent 
    
    :param space: discrete or continous space
    :param desired_type: Discrete_Space or Continous_Space
        which we want type(space) to match
    """
    if type(space) != desired_type:
        error(f"need space of type {desired_type}, but got space of type {type(space)}")
