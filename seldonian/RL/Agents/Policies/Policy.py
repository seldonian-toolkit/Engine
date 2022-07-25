class Policy:
    pass

class Discrete_Action_Policy(Policy):
    def __init__(self, min_action, num_actions):
        self.min_action = min_action #e.g., if environment actions are {-1, 0, 1}, then this is -1
        self.num_actions = num_actions

    def from_0_indexed_action_to_environment_action(self, action_0_indexed):
        return action_0_indexed + self.min_action

    def from_environment_action_to_0_indexed_action(self, env_action):
        return env_action - self.min_action
