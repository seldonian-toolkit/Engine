from seldonian.RL.environments.Environment import *
from seldonian.RL.Env_Description.Env_Description import *
from seldonian.RL.environments.mountaincar import *

class N_step_mountaincar(Environment):
    def __init__(self):
        self.n_steps = 20
        self.mc_env = Mountaincar()
        self.env_description = self.mc_env.create_env_description()
        self.terminal_state = False
        self.vis = False
        self.reset()

    def reset(self):
        self.mc_env.reset()
        self.terminal_state = self.mc_env.terminal_state

    def transition(self, action):
        reward = 0.0
        for _ in range(self.n_steps):
            reward += self.mc_env.transition(action)
            if self.mc_env.terminal_state:
                self.terminal_state = True
                break
        return reward

    def get_observation(self):
        return self.mc_env.get_observation()

    def visualize(self):
        self.mc_env.visualize()
