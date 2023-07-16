import autograd.numpy as np

###### Likelihhood Ratio Functions
def Mean_Squared_Error_Likelihood_Ratio(proposal, original, zhat, std):
    if std == 0:
        std = 1e-15
    # print(proposal, original, np.exp(- ((zhat - proposal) ** 2 - (zhat - original) ** 2).sum() / 2 / std ** 2))
    return np.exp(- ((zhat - proposal) ** 2 - (zhat - original) ** 2).sum() / 2 / std ** 2)

##### Likelihood Ratio Functions when also infer std
def Mean_Squared_Error_Likelihood_Ratio_Infer_Std(proposal, original, zhat):
    """
    :param proposal:
        Proposal parameters. [mean, std]
    :type List(float)

    :param original:
        Current parameters. [mean, std]
    :type List(float)
    """
    mean_prop, std_prop = proposal
    mean_orig, std_orig = original

    likelihood_ratio = np.exp(-(((zhat - mean_prop) / std_prop) ** 2 - ((zhat - mean_orig) / std_orig) ** 2).sum() / 2 + np.log(std_orig / std_prop) * len(zhat))
    # print(likelihood_ratio)
    return likelihood_ratio

################################
def get_likelihood_ratio(statistic_name, zhat, datasize, infer_std):
    """
    Function to get likelihood ratio functions from
    statistic name and zhat

    :param datasize:
        Size of safety data set.
    :type datasize: int

    :param infer_std:
        Whether to infer std
    :type infer_std: bool

    :ivar power:
        Ratio of size of safety data set
        to size of zhat
    :vartype power: float
    """

    power = (datasize / len(zhat))
    # print("Power:", power)

    if statistic_name == "Mean_Squared_Error":
        if infer_std:
            def wrapped_likelihood_ratio(proposal, original):
                return Mean_Squared_Error_Likelihood_Ratio_Infer_Std(proposal, original, zhat)
            return wrapped_likelihood_ratio
        else:
            # Use std of zhat as std of normal assumption
            # Scale std of zhat to predict what would be
            # the std for safety data (replace denominator
            # of variance with size of safety set instead of
            # size of candidate set)
            std = np.std(zhat) * np.sqrt(len(zhat) / datasize)
            # print(std)

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

