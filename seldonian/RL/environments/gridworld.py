from seldonian.RL.environments.Environment import *
from seldonian.RL.Env_Description.Env_Description import *

class Gridworld(Environment):
    def __init__(self, size):
        self.size = size
        self.num_states = size*size
        self.env_description = self.create_env_description(self.num_states)
        self.state = 0
        self.terminal_state = False
        self.time = 0
        self.max_time = 101
        self.vis = False
        self.gamma = 0.9

    def create_env_description(self, num_states):
        observation_space = Discrete_Space(0, num_states-1)
        action_space = Discrete_Space(0, 3)
        return Env_Description(observation_space, action_space)

    def reset(self):
        self.state = 0
        self.time = 0
        self.terminal_state = False

    def transition(self, action):
        reward = -1
        self.time += 1
        self.update_position(action)
        if self.vis:
            self.visualize()

        if self.is_in_goal_state() or self.time >= self.max_time - 1:
            self.terminal_state = True
            if self.is_in_goal_state():
                reward = 0
        return reward

    def get_observation(self):
        return self.state

    def update_position(self, action):
        if action == 0: #up
            if self.state >= self.size: #if not on top row
                self.state -= self.size
        elif action == 1: #right
            if (self.state + 1) % self.size != 0: #not on right column
                self.state += 1
        elif action == 2: #down
            if self.state < self.num_states - self.size: #not on bottom row
                self.state += self.size
        elif action == 3: #left
            if self.state % self.size != 0: #not on left column
                self.state -= 1
        else:
            raise Exception(f"invalid gridworld action {action}")

    def is_in_goal_state(self):
        return self.state == self.num_states - 1

    def visualize(self):
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


