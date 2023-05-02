

class PriorDistribution:

    def __init__(self, type, mean, width, infer_std):
        self.type = type
        self.mean = mean
        self.width = width
        self.infer_std = infer_std

    def prior_ratio(self, proposal, original):
        if  self.type == "jeffrey":
            if self.infer_std:
                # prior(mu, sigma) = 1 / sigma ** 2
                return (original[1] / proposal[1]) ** 2
            else:
                return 1
        elif self.type == "normal":
            raise NotImplementedError()
        else:
            raise NotImplementedError()