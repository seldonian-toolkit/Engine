from seldonian.RL.Agents.Function_Approximators.Function_Approximator import *

class Linear_FA(Function_Approximator):
    def __init__(self, basis, weights):
        self.basis = basis
        self.weights = weights


class Linear_state_action_value_FA(Linear_FA):
    def __init__(self, basis, weights):
        super().__init__(basis, weights)
        HERE, working on this and also need to add a basis