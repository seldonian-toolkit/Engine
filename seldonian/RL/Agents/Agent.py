class Agent(object):
    def __init__(self):
        """Base class for all RL agents. Override all methods
        below in child class implementation"""
        pass

    def choose_action(self, observation):
        """Choose an action given an observation. To be overridden

        :param observation: The current observation of the agent,
            type depends on environment.
        """
        raise NotImplementedError()

    def update(self, observation, next_observation, reward, terminated):
        """
        Updates agent's parameters according to the learning rule
        To be overriden

        :param observation: The current observation of the agent,
            type depends on environment.

        :param next_observation: The observation of the agent after
            an action is taken

        :param reward: The reward for taking the action

        :param terminated: Whether next_observation is a
            terminal observation
        :type terminated: bool
        """
        raise NotImplementedError()

    def get_prob_this_action(self, observation, action):
        """Get probability of a given action provided environment
        is in a observation. To be overridden

        :param observation: The current observation of the agent,
            type depends on environment.

        :param observation: The current action of the agent,
            type depends on environment.
        """
        raise NotImplementedError()

    def set_new_params(self, theta):
        """Update the parameters of the agent's policy to theta.

        :param theta: policy parameters
        """
        raise NotImplementedError()

    def get_params(self):
        """Retrieve the parameters of the agent's policy"""
        raise NotImplementedError()

    def get_policy(self):
        """Retrieve the agent's policy object"""
        raise NotImplementedError()
