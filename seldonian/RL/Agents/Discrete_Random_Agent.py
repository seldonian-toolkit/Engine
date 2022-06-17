from seldonian.RL.Agents.Agent import *
import numpy as np

class Discrete_Random_Agent(Agent):
    def __init__(self, env_description):
        self.min_action = env_description.action_space.min
        self.max_action = env_description.action_space.max

    def choose_action(self, observation):
        return np.random.randint(self.min_action, self.max_action + 1) #+1 because this function's high is exclusive

    def update(self, observation, next_observation, reward, terminated):
        pass
