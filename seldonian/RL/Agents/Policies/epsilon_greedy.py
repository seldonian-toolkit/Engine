# This file/class is partially finished, but was probably a mistake to make because non-differentiable.  Going to leave it undeleted for now in case it's useful, but if significant time has gone by (currently 7/22/22), feel free to delete this file

# from seldonian.RL.Agents.Policies.Policy import *
# import numpy as np
# from utils import *
#
# class Epsilon_Greedy(Discrete_Action_Policy):
#     def __init__(self, epsilon, min_action, num_actions):
#         super().__init__(min_action, num_actions)
#         self.epsilon = epsilon
#
#     def get_greedy_action_0_indexed(self, action_values):
#         max_actions_zero_indexed = argmax_multi(action_values)
#         return np.random.choice(max_actions_zero_indexed) #if there is a tie, pick one at random
#
#     def choose_action(self, action_values):
#         if len(action_values) != self.num_actions:
#             error(f"should have {self.num_actions}, but got {len(action_values)} action values")
#         if np.random.rand() < self.epsilon:
#             action_0_indexed = self.get_random_action_0_indexed()
#         else:
#             action_0_indexed = self.get_greedy_action_0_indexed(action_values)
#         return self.from_0_indexed_action_to_environment_action(action_0_indexed)
#
#     def get_random_action_0_indexed(self):
#         return np.random.choice(range(self.num_actions))
#
# # def main():
# #     egreedy = Epsilon_Greedy(0.1, -1, 3)
# #     my_array = np.array([-.5, -.8, -54.])
# #     for _ in range(111):
# #         choice = egreedy.choose_action(my_array)
# #         print(choice)
# #
# #
# # main()
