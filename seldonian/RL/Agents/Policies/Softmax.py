from seldonian.RL.Agents.Policies.Policy import *
import autograd.numpy as np
from seldonian.utils.RL_utils import *

class Softmax(Discrete_Action_Policy):
    def __init__(self, min_action, num_actions):
        super().__init__(min_action, num_actions)

    def choose_action(self, action_values):
        if len(action_values) != self.num_actions:
            error(f"should have {self.num_actions} actions, but got {len(action_values)} action values")

        action_probs = self.get_action_probs_from_action_values(action_values)

        roulette_wheel_start = 0.0
        stop_value = np.random.rand()
        for action_num_zero_indexed in range(self.num_actions):
            roulette_wheel_start += action_probs[action_num_zero_indexed]
            if roulette_wheel_start >= stop_value:
                return self.from_0_indexed_action_to_environment_action(action_num_zero_indexed)

        print(stop_value)
        print(roulette_wheel_start)
        print(action_probs)
        error("reached the end of SoftMax.choose_action(), this should never happen")

    def get_action_probs_from_action_values(self, action_values):
        e_to_the_something_terms = self.get_e_to_the_something_terms(action_values)
        denom = sum(e_to_the_something_terms)
        return e_to_the_something_terms / denom

    def get_e_to_the_something_terms(self, action_values):
        max_value = np.max(action_values)
        e_to_the_something_terms = np.exp(action_values - max_value) #subtract max for numerical stability
        return e_to_the_something_terms
