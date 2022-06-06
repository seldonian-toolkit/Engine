import os,sys
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from seldonian.seldonian_algorithm import seldonian_algorithm
from seldonian.utils.io_utils import load_pickle

if __name__ == '__main__':
	# gpa dataset
	interface_output_dir = os.path.join('/Users/ahoag/beri/code',
		'interface_outputs/mountain_car_interface_withspec')
	specfile = os.path.join(interface_output_dir,'spec.pkl')
	spec = load_pickle(specfile)
	spec.optimization_hyperparams['num_iters'] = 20
	spec.optimization_hyperparams['alpha_theta'] = 0.05
	spec.optimization_hyperparams['alpha_lamb'] = 0.05
	passed_safety,candidate_solution = seldonian_algorithm(spec)
	print(passed_safety,candidate_solution)
