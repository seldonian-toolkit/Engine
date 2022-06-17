from seldonian.RL.Env_Description.Spaces import *

class Env_Description:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space