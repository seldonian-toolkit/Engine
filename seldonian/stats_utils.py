import numpy as np
from scipy.stats import t

def stddev(v):
    """
    Sample standard deviation of the vector v,
    with Bessel's correction
    """
    return np.std(v,ddof=1)         

def tinv(p, nu):
    """
    Returns the inverse of Student's t CDF 
    using the degrees of freedom in nu for the corresponding
    probabilities in p. 
    Python implementation of Matlab's tinv function:
     https://www.mathworks.com/help/stats/tinv.html
    """
    return t.ppf(p, nu)

def weighted_sum_gamma(arr,gamma=0.9):
    """ Calculate weighted sum of arr,
    where weights are gamma**(index of arr)  """
    weights = np.power(gamma,range(len(arr)))
    return np.average(arr,weights=weights)*np.sum(weights)