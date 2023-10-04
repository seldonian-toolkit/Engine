from seldonian.RL.Agents.Agent import *
from seldonian.RL.Agents.Function_Approximators.Table import *
from seldonian.RL.Agents.Policies.Softmax import *
from seldonian.RL.Agents.Policies.SimglucosePolicyFixedArea import SigmoidPolicyFixedArea

class SimglucoseFixedAreaAgent(Agent):
    def __init__(self, bb_crmin, bb_crmax, bb_cfmin, bb_cfmax, cr_shrink_factor, cf_shrink_factor):
        """

        :param hyperparameter_and_setting_dict: Specifies the
            environment, agent, number of episodes per trial,
            and number of trials
        :param env_description: an object for accessing attributes
            of the environment
        :type env_description: :py:class:`.Env_Description`
        """
        self.policy = SigmoidPolicyFixedArea(bb_crmin, bb_crmax, bb_cfmin, bb_cfmax, cr_shrink_factor, cf_shrink_factor)

    def choose_action(self, obs):
        """Return a CR,CF by sampling from uniform random distributions
        whose bounds are determined by the crmin,crmax,cfmin,cfmax which
        are determined from sigmoiding the theta values (policy weights).

        :param obs: The current observation of the agent,
                type depends on environment
        :return: array of actions
        """
        theta = self.policy.get_params()
        cr1,cr2,cf1,cf2 = self.policy.theta2crcf(theta)
        cr = np.random.uniform(cr1,cr2)
        cf = np.random.uniform(cf1,cf2)
        return cr,cf

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