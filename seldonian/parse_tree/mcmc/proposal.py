import numpy as np

class ProposalDistribution:
    """
    Normal proposal distribution
    """
    def __init__(self, proposal_width, infer_std):
        self.proposal_width = proposal_width
        self.infer_std = infer_std

    def propose(self, x):
        # supports multidimension x
        proposal = np.random.normal(loc=x, scale=self.proposal_width)

        if self.infer_std:
            # make sure std > 0
            if proposal[1] <= 0:
                proposal[1] = 1e-10
        
        return proposal