import autograd.numpy as np   # Thinly-wrapped version of Numpy
from scipy.stats import t

def stddev(v):
    """
    Sample standard deviation of the vector v,
    with Bessel's correction

    :param v: vector of data
    :type v: Numpy ndarray

    :return: Standard deviation with Bessel's correction
    :rtype: float
    """
    return np.std(v,ddof=1)         

def tinv(p, nu):
    """
    Returns the inverse of Student's t CDF 
    using the degrees of freedom in nu for the corresponding
    probabilities in p. 
    
    Python implementation of Matlab's tinv function:
    https://www.mathworks.com/help/stats/tinv.html

    :param p: Probability
    :type p: float

    :param nu: Degrees of freedom
    :type nu: int

    :return: Inverse of the Student's t CDF
    :rtype: float
    """
    return t.ppf(p, nu)

def weighted_sum_gamma(arr,gamma=0.9):
    """ Calculate weighted sum of an array,
    where weights are gamma**(index of arr). 
    Used in calculating sum of weighted rewards in RL

    :param arr: An input array
    :type arr: Numpy ndarray

    :param gamma: The constant used for weighting the array
    :type gamma: float  

    :return: The weighted sum
    :rtype: float  
    """
    weights = np.power(gamma,range(len(arr)))
    return np.average(arr,weights=weights)*np.sum(weights)

def generate_data(numPoints,loc_X=0.0,loc_Y=0.0,sigma_X=1.0,sigma_Y=1.0):
    """ Generate 2D random normal data
    
    :param numPoints: The number of data points to generate
    :type numPoints: int

    :param loc_X: The center of the normal distribution 
        in the X dimension
    :type loc_X: float

    :param loc_Y: The center of the normal distribution 
        in the Y dimension
    :type loc_Y: float

    :param sigma_X: The standard deviation of the normal distribution 
        in the X dimension
    :type sigma_X: float

    :param sigma_Y: The standard deviation of the normal distribution 
        in the Y dimension
    :type sigma_Y: float
    """

    X =     np.random.normal(loc_X, sigma_X, numPoints) # Sample x from a standard normal distribution
    Y = X + np.random.normal(loc_Y, sigma_Y, numPoints) # Set y to be x, plus noise from a standard normal distribution
    return (X,Y)