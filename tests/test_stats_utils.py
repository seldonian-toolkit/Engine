import pytest
import autograd.numpy as np

from seldonian.utils.stats_utils import (stddev,
	tinv,weighted_sum_gamma)

### Begin tests

def test_stddev():
	""" Test the Bessel's-corrected standard deviation 
	function """

	arr = np.array([1,2,3])
	assert stddev(arr) == 1.0

	arr2 = [1,2,3]
	assert stddev(arr) == 1.0

	arr3 = np.array([1.0,2.0,3.0])
	assert stddev(arr3) == 1.0

def test_tinv():
	""" Test the tinv() function """
	
	assert tinv(0.95,1000) == pytest.approx(1.646379)
	assert tinv(0.1, 1000) == pytest.approx(-1.282399)

def test_weighted_sum_gamma():
	""" Test the function to calculate the 
	weighted sum where the weights are gamma**i for i in range(0,len(array)) """
	
	arr = np.array([1.0])
	assert weighted_sum_gamma(arr,gamma=0.9) == 1.0

	arr = np.array([1,2,3])
	assert weighted_sum_gamma(arr,gamma=0.9) == 5.23

	arr = np.arange(1000)
	assert weighted_sum_gamma(arr,gamma=0.9) == pytest.approx(90.0)

	arr = np.arange(-1000,0)
	assert weighted_sum_gamma(arr,gamma=0.9) == pytest.approx(-9910.0)

	arr=[float('-inf'),1]
	assert weighted_sum_gamma(arr,gamma=0.9) == float('-inf')

	arr=[float('inf'),1]
	assert weighted_sum_gamma(arr,gamma=0.9) == float('inf')

	arr=[float('-inf'),float('inf')]
	assert np.isnan(weighted_sum_gamma(arr,gamma=0.9))