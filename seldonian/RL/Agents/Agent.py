from seldonian.RL.Env_Description.Env_Description import *
from seldonian.RL.Agents.Function_Approximators.Table import *
from seldonian.RL.Agents.Function_Approximators.Linear_FA import *
from seldonian.RL.Agents.Bases.Fourier import *

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

    def make_state_action_FA(self, env_description, hyperparam_and_setting_dict):
        """ Create a function approximator from an environment description and
        dictionary specification

        :param env_description: an object for accessing attributes
            of the environment
        :type env_description: :py:class:`.Env_Description`

        :param hyperparameter_and_setting_dict: Specifies the
            environment, agent, number of episodes per trial,
            and number of trials

        :return: function approximator, type depends on whether observation 
            space is discrete or continous
        """
        if type(env_description.observation_space) == Discrete_Space and type(env_description.action_space) == Discrete_Space:
            return construct_Q_Table_From_Env_Description(env_description)
        if type(env_description.observation_space) == Continuous_Space and type(env_description.action_space) == Discrete_Space:
            return self.construct_basis_and_linear_FA(env_description, hyperparam_and_setting_dict)
        else:
            error(f"unhandled state type {type(env_description.observation_space)} and action type {type(env_description.action_space)} for make_state_action_FA()")

    def construct_basis_and_linear_FA(self, env_description, hyperparam_and_setting_dict):
        """Create a basis and linear function approximator 
        from an environment description and dictionary specification

        :param env_description: an object for accessing attributes
            of the environment
        :type env_description: :py:class:`.Env_Description`

        :param hyperparameter_and_setting_dict: Specifies the
            environment, agent, number of episodes per trial,
            and number of trials
        
        """
        basis_type = hyperparam_and_setting_dict["basis"]
        if basis_type == "Fourier":
            basis = Fourier(hyperparam_and_setting_dict, env_description)
        else:
            error("unknown basis type ", basis_type)
        return Linear_state_action_value_FA(basis, env_description)

    def get_params(self):
        raise NotImplementedError()
