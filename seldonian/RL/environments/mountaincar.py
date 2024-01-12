from seldonian.RL.environments.Environment import *
from seldonian.RL.Env_Description.Env_Description import *
from math import cos
from seldonian.utils.RL_utils import *


class Mountaincar(Environment):
    def __init__(self):
        """Classic Mountaincar environment with hardcoded position and velocity bounds.
        Actions: -1,0,1 -> force left, no force, force right.

        :ivar env_description: contains attributes describing the environment
        :vartype env_description: :py:class:`.Env_Description`
        :ivar terminal_state: Whether the terminal obs is occupied
        :vartype terminal_state: bool
        :ivar time: The current timestep
        :vartype time: int
        :ivar position: The 1D physical position of the car, initialized at -0.5.
        :vartype position: float
        :ivar velocity: The 1D velocity of the car, initialized at 0.0.
        :vartype velocity: float
        :ivar max_time: Maximum allowed timestep
        :vartype max_time: int
        """
        self.env_description = self.create_env_description()
        self.terminal_state = False
        self.time = 0
        self.position = -0.5
        self.velocity = 0.0
        self.max_time = 1000
        self.vis = False
        self.reset()

    def create_env_description(self):
        """Creates the environment description object.

        :param num_states: The number of states

        :return: Environment description for the obs and action spaces
        :rtype: :py:class:`.Env_Description`
        """
        state_space_bounds = np.array([[-1.2, 0.5], [-0.07, 0.07]])
        state_space = Continuous_Space(state_space_bounds)
        action_space = Discrete_Space(-1, 1)
        return Env_Description(state_space, action_space)

    def reset(self):
        """Go back to initial obs and timestep"""
        self.position = -0.5
        self.velocity = 0.0
        self.time = 0
        self.terminal_state = False

    def transition(self, action):
        """Transition between states given an action, return a reward.

        :param action: A possible action at the current obs
        :return: reward for reaching the next obs
        """
        self.check_valid_mc_action(action)
        self.time += 1

        self.update_velocity(action)
        self.position_and_termination_update()

        reward = -1.0
        if self.terminal_state:
            reward = 0.0
        return reward

    def update_velocity(self, action):
        """Apply the velocity update rule

        :param action: A possible action at the current obs
        """
        self.velocity += 0.001 * action - 0.0025 * cos(3.0 * self.position)
        self.velocity = clamp(
            self.velocity,
            self.env_description.observation_space.bounds[1][0],
            self.env_description.observation_space.bounds[1][1],
        )

    def position_and_termination_update(self):
        """Update the position given the current velocity.
        Check to see if we have gone outside position bounds.
        Also check to see if we have reached the goal position.
        """
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
        """Checks to ensure a valid action was taken.

        :param action: A proposed action at the current obs
        """
        if action != -1 and action != 0 and action != 1:
            raise (Exception(f"invalid action {action}"))

    def get_observation(self):
        """Get the position and velocity at the current timestep"""
        return np.array([self.position, self.velocity])
