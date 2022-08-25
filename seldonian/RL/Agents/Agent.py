
class Agent:
    """ Base class for all RL agents. Many methods
    require overriding in child classes to be used """
    def choose_action(self, observation):
        """ Choose an action given an observation. To be overridden 

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

        :param terminated: Whether next_observation is the 
            terminal observation
        :type terminated: bool
        """
        raise NotImplementedError()

    def get_prob_this_action(self, observation, action):
        """ Get probability of a given action provided environment 
        is in a observation. To be overridden 

        :param observation: The current observation of the agent,
            type depends on environment.
        """
        raise NotImplementedError()

    def set_new_params(self, theta):
        raise NotImplementedError()

    def get_params(self):
        raise NotImplementedError()

    def get_policy(self):
        raise NotImplementedError()
