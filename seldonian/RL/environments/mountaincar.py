from seldonian.RL.environments.Environment import *
from seldonian.RL.Env_Description.Env_Description import *
from math import cos
from seldonian.utils.RL_utils import *

class Mountaincar(Environment):
    def __init__(self):
        self.env_description = self.create_env_description()
        self.terminal_state = False
        self.time = 0
        self.position = -.5
        self.velocity = 0.0
        self.max_time = 3000
        self.vis = False
        self.reset()

    def create_env_description(self):
        state_space_bounds = np.array([[-1.2, 0.5], [-.07, .07]])
        state_space = Continuous_Space(state_space_bounds)
        action_space = Discrete_Space(-1, 1)
        return Env_Description(state_space, action_space)

    def reset(self):
        self.position = -.5
        self.velocity = 0.0
        self.time = 0
        self.terminal_state = False

    def transition(self, action):
        self.check_valid_mc_action(action)
        self.time += 1

        self.update_velocity(action)
        self.position_and_termination_update()

        reward = -1.0
        if self.terminal_state:
            reward = 0.0
        return reward

    def update_velocity(self, action):
        self.velocity += .001 * action - .0025 * cos(3.0 * self.position)
        self.velocity = clamp(self.velocity, self.env_description.observation_space.bounds[1][0], self.env_description.observation_space.bounds[1][1])

    def position_and_termination_update(self):
        self.position += self.velocity
        pos_lower_bound = self.env_description.observation_space.bounds[0][0]
        pos_upper_bound = self.env_description.observation_space.bounds[0][1]
        if self.position <= pos_lower_bound:
            self.position = pos_lower_bound
            self.velocity = 0
        elif self.position >= pos_upper_bound:
            self.terminal_state = True
        if self.time >= self.max_time:
            self.terminal_state = True

    def visualize(self):
        error("mountain car visualize method not implemented")

    def check_valid_mc_action(self, action):
        if action != -1 and action != 0 and action != 1:
            raise(Exception(f"invalid action {action}"))

    def get_observation(self):
        return np.array([self.position, self.velocity])
