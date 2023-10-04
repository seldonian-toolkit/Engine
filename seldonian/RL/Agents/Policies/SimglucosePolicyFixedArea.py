import autograd.numpy as np

from seldonian.RL.Agents.Policies.Policy import Policy
from seldonian.RL.Agents.Function_Approximators.Function_Approximator import Function_Approximator


class SigmoidPolicyFixedArea(Policy):
    def __init__(self, 
        bb_crmin, 
        bb_crmax, 
        bb_cfmin, 
        bb_cfmax, 
        cr_shrink_factor, 
        cf_shrink_factor):
        """
        
        """
        self.bb_crmin = bb_crmin
        self.bb_crmax = bb_crmax
        self.bb_cfmin = bb_cfmin
        self.bb_cfmax = bb_cfmax

        self.cr_shrink_factor = cr_shrink_factor
        self.crwidth = (self.bb_crmax-self.bb_crmin)/cr_shrink_factor
        self.new_crmin = self.bb_crmin+self.crwidth/2
        # center can range from crmin to new_crmax
        self.new_cr_range = (self.bb_crmax - self.bb_crmin) - self.crwidth

        self.cf_shrink_factor = cf_shrink_factor
        self.cfwidth = (self.bb_cfmax-self.bb_cfmin)/cf_shrink_factor
        self.new_cfmin = self.bb_cfmin+self.cfwidth/2
        # center can range from cfmin to new_cfmax
        self.new_cf_range = (self.bb_cfmax - self.bb_cfmin) - self.cfwidth

        self.FA = Function_Approximator() # needed to get and set self.FA.weights 
        # Initialize weights
        self.FA.weights = np.array([-5.0,5.0]) # each can range -inf to +inf

    def set_new_params(self, new_params):
        """Set the parameters of the agent

        :param new_params: array of weights
        """
        self.FA.set_new_params(new_params)

    def get_params(self):
        """Get the current parameters (weights) of the agent

        :return: array of weights
        """
        return self.FA.weights

    def theta2crcf(self, theta):
        """Take theta and return cr_center,cf_center
        Keeping in mind that they must be capped 
        so that when the fixed width of the box
        is added, it doesn't exceed the original bounds.
        """
        # if no shrinking then just use the entire original box
        if self.cr_shrink_factor <= 1:
            cr1 = self.bb_crmin
            cr2 = self.bb_crmax
        else:
            cr_cen = self._sigmoid(theta[0])*self.new_cr_range+self.new_crmin
            cr1 = cr_cen - self.crwidth/2
            cr2 = cr1 + self.crwidth
        # if no shrinking then just use the entire original box
        if self.cf_shrink_factor <= 1:
            cf1 = self.bb_cfmin
            cf2 = self.bb_cfmax
        else:
            cf_cen = self._sigmoid(theta[1])*self.new_cf_range+self.new_cfmin
            cf1 = cf_cen - self.cfwidth/2
            cf2 = cf1 + self.cfwidth
        return cr1,cr2,cf1,cf2
    
    def _sigmoid(self, X, steepness=1.0):
        return 1 / (1 + np.exp(-steepness*X))

    
