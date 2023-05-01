import numpy as np

class ProposalDistribution:
    """
    Normal proposal distribution
    """
    def __init__(self, proposal_width):
        self.proposal_width = proposal_width

    def propose(self, x):
        return np.random.normal(loc=x, scale=self.proposal_width)