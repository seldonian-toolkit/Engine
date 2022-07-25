from seldonian.RL.Agents.Agent import *
import autograd.numpy as np

class Mountain_car_rough_solution(Agent):
    def choose_action(self, observation):
        if observation[1] < 0:
            return -1
        else:
            return 1

    def update(self, observation, next_observation, reward, terminated):
        pass

    def get_prob_this_action(self, observation, action):
        return 1.0