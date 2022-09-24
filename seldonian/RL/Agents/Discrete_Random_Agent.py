from seldonian.RL.Agents.Agent import *
import autograd.numpy as np

class Discrete_Random_Agent(Agent):
    def __init__(self, env_description):
        """ An agent that acts on discrete 
        observation and action spaces.
        Picks actions according to uniform random policy. 
        Is not capable of learning.

        :param env_description: an object for accessing attributes
            of the environment
        :type env_description: :py:class:`.Env_Description`
        """
        self.min_action = env_description.action_space.min
        self.max_action = env_description.action_space.max
        self.num_actions = self.max_action - self.min_action + 1

    def choose_action(self, observation):
        return np.random.randint(self.min_action, self.max_action + 1) #+1 because this function's high is exclusive

    def update(self, observation, next_observation, reward, terminated):
        pass

    def get_prob_this_action(self, observation, action):
        return 1.0 / self.num_actions
