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
        self.FA.weights = np.array([0.0,0.0,0.0,0.0]) 

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

    def renorm_weights_action(self, theta):
        """Reshape theta's cr and cf bounding box to be within the bounding box 
        set at the outside, specified by the self.bb_crmin, etc.

        """
        
        cr1,cr2,cf1,cf2 = theta
        # First make sure that cr1<cr2 and cf1<cf2
        if cr1 >= cr2: cr1,cr2 = cr2,cr1
        if cf1 >= cf2: cf1,cf2 = cf2,cf1

        cr_size = self.bb_crmax - self.bb_crmin
        cf_size = self.bb_cfmax - self.bb_cfmin
        if cr1<self.bb_crmin:
            theta[0] = self.bb_crmin + self._sigmoid(cr1-self.bb_crmin)*cr_size
        if cr2>self.bb_crmax:
            theta[1] = self.bb_crmin + self._sigmoid(cr2-self.bb_crmin)*cr_size
        
        if cf1<self.bb_cfmin:
            theta[2] = self.bb_cfmin + self._sigmoid(cf1-self.bb_cfmin)*cf_size

        if cf2>self.bb_cfmax:
            theta[3] = self.bb_cfmin + self._sigmoid(cf2-self.bb_cfmin)*cf_size
        return theta
    
    def _sigmoid(self, X, steepness=100):
        return 1 / (1 + np.exp(-steepness*X))

    
