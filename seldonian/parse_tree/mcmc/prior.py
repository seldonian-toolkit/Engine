

class PriorDistribution:

    def __init__(self, type, mean, width):
        self.type = type
        self.mean = mean
        self.width = width

    def prior_ratio(self, proposal, original):
        if  self.type == "jeffrey":
            return 1
        elif self.type == "normal":
            raise NotImplementedError()
        else:
            raise NotImplementedError()