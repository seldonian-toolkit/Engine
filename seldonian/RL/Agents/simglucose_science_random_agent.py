from seldonian.RL.Agents.Agent import *
from seldonian.RL.Agents.Function_Approximators.Table import *
from seldonian.RL.Agents.Policies.Softmax import *
from seldonian.RL.Agents.Policies.SimglucosePolicy import ScienceSigmoidPolicy

class SimglucoseScienceAgent(Agent):
    def __init__(self, bb_crmin, bb_crmax, bb_cfmin, bb_cfmax):
        """

        :param hyperparameter_and_setting_dict: Specifies the
            environment, agent, number of episodes per trial,
            and number of trials
        :param env_description: an object for accessing attributes
            of the environment
        :type env_description: :py:class:`.Env_Description`
        """
        self.policy = ScienceSigmoidPolicy(bb_crmin, bb_crmax, bb_cfmin, bb_cfmax)

    def choose_action(self, obs):
        """Return a CR,CF tuple by sampling from uniform random distributions

        :param obs: The current observation of the agent,
                type depends on environment
        :return: array of actions
        """

        return (
            np.random.uniform(
                self.policy.FA.weights[0],
                self.policy.FA.weights[1]),
            np.random.uniform(
                self.policy.FA.weights[2],
                self.policy.FA.weights[3])
        ) 

    def get_prob_this_action(self,observation, action):
        return 0 # this is a continuous action space so all individual actions have 0 probability

    def update(self, observation, next_observation, reward, terminated):
        """
        Noop, but it must be implemented

        :param observation: The current observation of the agent,
                type depends on environment.
        :param next_observation: The observation of the agent after
                an action is taken
        :param reward: The reward for taking the action
        :param terminated: Whether next_observation is the
                terminal observation
        :type terminated: bool
        """
        pass

    def set_new_params(self, new_params):
        """Set the parameters of the agent

        :param new_params: array of weights
        """
        self.policy.set_new_params(new_params)