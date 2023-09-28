class Environment(object):
    """Base class for all RL environments"""

    def get_observation(self):
        """Get current observation.
        Override this method in child class implementation
        """
        raise NotImplementedError()

    def transition(self, action):
        """Transition to a new observation given an action

        :param action: A possible action in the environment
        Override this method in child class implementation
        """
        raise NotImplementedError()

    def reset(self):
        """Reset observation to initial observation
        Override this method in child class implementation
        """
        raise NotImplementedError()

    def get_env_description(self):
        """Get environment description
        Override this method in child class implementation
        """
        return self.env_description

    def terminated(self):
        """Get the terminal obs"""
        return self.terminal_state

    def visualize(self):
        """Print out current observation, useful for debugging
        Override this method in child class implementation
        """
        raise NotImplementedError()

    def start_visualizing(self):
        """Turn on visualization debugger"""
        self.vis = True

    def stop_visualizing(self):
        """Turn off visualization debugger"""
        self.vis = False
