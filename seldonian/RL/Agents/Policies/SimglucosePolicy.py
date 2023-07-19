import autograd.numpy as np

from seldonian.RL.Agents.Policies.Policy import Policy
from seldonian.RL.Agents.Function_Approximators.Function_Approximator import Function_Approximator


class ScienceSigmoidPolicy(Policy):
    def __init__(self, 
        bb_crmin, bb_crmax, bb_cfmin, bb_cfmax):
        """
        
        """
        self.bb_crmin = bb_crmin
        self.bb_crmax = bb_crmax
        self.bb_cfmin = bb_cfmin
        self.bb_cfmax = bb_cfmax

        self.FA = Function_Approximator() # needed to get and set self.FA.weights 
        # Initialize weights to the large physician approved bounding box
        self.FA.weights = np.array([0.0,0.0,0.0,0.0]) # each can range -inf to +inf

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
        """Take theta and return crmin,crmax,cfmin,cfmax
        """
        
        
        cr_size = self.bb_crmax - self.bb_crmin
        cf_size = self.bb_cfmax - self.bb_cfmin
        cr1 = self._sigmoid(theta[0]) * cr_size + self.bb_crmin
        cr2 = self._sigmoid(theta[1]) * cr_size + self.bb_crmin
        cr1,cr2 = min(cr1,cr2),max(cr1,cr2)

        cf1 = self._sigmoid(theta[2]) * cf_size + self.bb_cfmin
        cf2 = self._sigmoid(theta[3]) * cf_size + self.bb_cfmin
        cf1,cf2 = min(cf1,cf2),max(cf1,cf2)

        return cr1,cr2,cf1,cf2
    
    def _sigmoid(self, X, steepness=1.0):
        return 1 / (1 + np.exp(-steepness*X))

    
