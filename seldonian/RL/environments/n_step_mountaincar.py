from seldonian.RL.environments.Environment import *
from seldonian.RL.Env_Description.Env_Description import *
from seldonian.RL.environments.mountaincar import *


class N_step_mountaincar(Environment):
    def __init__(self):
        """Extends the classic Mountaincar environment such that
        each action is taken n_steps times instead of once. Refer
        to :py:class:`.mountaincar.Mountaincar` docstring.

        :ivar n_steps: The number of repeated steps to take following
            a single action.
        :vartype n_steps: int
        :ivar mc_env: An instance of the Mountaincar class.
        :ivar env_description: contains attributes describing the environment
        :vartype env_description: :py:class:`.Env_Description`
        :ivar terminal_state: Whether the terminal obs is occupied
        :vartype terminal_state: bool
        :ivar gamma: The discount factor, hardcoded to 1.
        """
        self.n_steps = 20
        self.mc_env = Mountaincar()
        self.env_description = self.mc_env.create_env_description()
        self.terminal_state = False
        self.vis = False
        self.gamma = 1.0
        self.reset()

    def reset(self):
        """Go back to initial obs and timestep"""
        self.mc_env.reset()
        self.terminal_state = self.mc_env.terminal_state

    def transition(self, action):
        """Transition between states given an action, return a reward.

        :param action: A possible action at the current obs
        :return: reward for reaching the next obs
        """
        reward = 0.0
        for _ in range(self.n_steps):
            reward += self.mc_env.transition(action)
            if self.mc_env.terminal_state:
                self.terminal_state = True
                break
        return reward

    def get_observation(self):
        """Wrapper to get the position and velocity of the cart
        from the Mountain car environment.
        """
        return self.mc_env.get_observation()

    def visualize(self):
        self.mc_env.visualize()
