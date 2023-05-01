import autograd.numpy as np

##### Likelihhood Ratio Functions
def Mean_Squared_Error_Likelihood_Ratio(proposal, original, zhat, std):
    if std == 0:
        std = 1e-15
    return np.exp(- ((zhat - proposal) ** 2 - (zhat - original) ** 2).sum() / 2 / std ** 2)


################################
def get_likelihood_ratio(statistic_name, zhat, datasize):
    """
    Function to get likelihood ratio functions from
    statistic name and zhat

    :param datasize:
        Size of safety data set.
    :type dataset: int

    :ivar power:
        Ratio of size of safety data set
        to size of zhat
    :vartype power: float
    """

    power = (datasize / len(zhat))
    # print("Power:", power)

    if statistic_name == "Mean_Squared_Error":
        # use std of zhat as std of normal assumption
        std = np.std(zhat)

        # Wrap likelihood ratio function such that
        # it only takes proposal g and original (current) g
        # as inputs. 
        def wrapped_likelihood_ratio(proposal, original):
            # If in candidate selection, scale likelihood ratio
            # according to size of safety data set.
            # If in safety test, power is 1.
            return Mean_Squared_Error_Likelihood_Ratio(proposal, original, zhat, std) ** power
        
        return wrapped_likelihood_ratio
    
    raise NotImplementedError()

