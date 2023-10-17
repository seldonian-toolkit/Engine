from seldonian.RL.environments.Environment import *
from seldonian.RL.Env_Description.Env_Description import *


class Gridworld(Environment):
    def __init__(self, size=3):
        """Square gridworld RL environment of arbitrary size.
        Actions: 0,1,2,3 -> up,right,down,left

        :param size: The number of grid cells on a side
        :ivar num_states: The number of distinct grid cells
        :ivar env_description: contains attributes describing the environment
        :vartype env_description: :py:class:`.Env_Description`
        :ivar obs: The current obs
        :vartype obs: int
        :ivar terminal_state: Whether the terminal obs is occupied
        :vartype terminal_state: bool
        :ivar time: The current timestep
        :vartype time: int
        :ivar max_time: Maximum allowed timestep
        :vartype max_time: int
        :ivar gamma: The discount factor in calculating the expected return
        :vartype gamma: float
        """
        self.size = size
        self.num_states = size * size
        self.env_description = self.create_env_description(self.num_states)
        self.state = 0
        self.terminal_state = False
        self.time = 0
        self.max_time = 101
        # vis is a flag for visual debugging during obs transitions
        self.vis = False
        self.gamma = 0.9

    def create_env_description(self, num_states):
        """Creates the environment description object.

        :param num_states: The number of states
        :return: Environment description for the obs and action spaces
        :rtype: :py:class:`.Env_Description`
        """
        observation_space = Discrete_Space(0, num_states - 1)
        action_space = Discrete_Space(0, 3)
        return Env_Description(observation_space, action_space)

    def reset(self):
        """Go back to initial obs and timestep"""
        self.state = 0
        self.time = 0
        self.terminal_state = False

    def transition(self, action):
        """Transition between states given an action, return a reward.

        :param action: A possible action at the current obs
        :return: reward for reaching the next obs
        """
        reward = 0
        self.time += 1
        self.update_position(action)

        if self.is_in_goal_state() or self.time >= self.max_time - 1:
            self.terminal_state = True
            if self.is_in_goal_state():
                reward = 1
        if self.state == 7:
            reward = -1
        if self.vis:
            self.visualize()
            print("reward", reward)
        return reward

    def get_observation(self):
        """Get the current obs"""
        return self.state

    def update_position(self, action):
        """Helper function for transition() that updates the
        current position given an action

        :param action: A possible action at the current obs
        """
        if action == 0:  # up
            if self.state >= self.size:  # if not on top row
                self.state -= self.size
        elif action == 1:  # right
            if (self.state + 1) % self.size != 0:  # not on right column
                self.state += 1
        elif action == 2:  # down
            if self.state < self.num_states - self.size:  # not on bottom row
                self.state += self.size
        elif action == 3:  # left
            if self.state % self.size != 0:  # not on left column
                self.state -= 1
        else:
            raise Exception(f"invalid gridworld action {action}")

    def is_in_goal_state(self):
        """Check whether current obs is goal obs

        :return: True if obs is in goal obs, False if not
        """
        return self.state == self.num_states - 1

    def visualize(self):
        """Print out current obs information"""
        print_state = 0
        for y in range(self.size):
            for x in range(self.size):
                if print_state == self.state:
                    print("A", end="")
                else:
                    print("X", end="")
                print_state += 1
            print()
        print()
