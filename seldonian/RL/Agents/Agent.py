class Agent:
    def choose_action(self, observation):
        raise NotImplementedError()

    def update(self, observation, next_observation, reward, terminated):
        raise NotImplementedError()