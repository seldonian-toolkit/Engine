from seldonian.RL.Agents.Agent import *

class Keyboard_gridworld(Agent):
    def __init__(self, env_description):
        """ An agent used for debugging the gridworld
        environment. Not intended for public use. """
        pass

    def choose_action(self, observation):
        action = self.ask_for_input()
        while action not in [1, 2, 3, 4]:
            print("invalid action, please try again")
            action = self.ask_for_input()
        return action - 1 #env is zero-indexed

    def ask_for_input(self):
        return int(input("Enter 1, 2, 3, or 4 for UP, RIGHT, DOWN, LEFT respectively"))

    def update(self, observation, next_observation, reward, terminated):
        pass

    def get_prob_this_action(self, observation, action):
        return float("NaN")
