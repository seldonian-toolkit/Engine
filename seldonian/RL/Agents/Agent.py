class Agent:
    def choose_action(self, observation):
        raise NotImplementedError()

    def update(self, observation, next_observation, reward, terminated):
        raise NotImplementedError()

    def get_prob_this_action(self, observation, action):
        raise NotImplementedError()