class Environment:
    def get_observation(self):
        raise NotImplementedError()

    def transition(self, action):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def get_env_description(self):
        return self.env_description

    def terminated(self):
        return self.terminal_state

    def visualize(self):
        raise NotImplementedError()

    def start_visualizing(self):
        self.vis = True

    def stop_visualizing(self):
        self.vis = False