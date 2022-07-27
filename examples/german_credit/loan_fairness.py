import os

from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.io_utils import load_pickle

if __name__ == '__main__':
	# Load loan spec file
	specfile = './spec.pkl'
	spec = load_pickle(specfile)
	
	spec.use_builtin_primary_gradient_fn = False
	spec.optimization_hyperparams['alpha_theta'] = 0.01
	spec.optimization_hyperparams['alpha_lamb'] = 0.01
	spec.optimization_hyperparams['num_iters'] = 1000
	SA = SeldonianAlgorithm(spec)
	passed_safety,solution = SA.run()
	if passed_safety:
		print("Passed safety test!")
	else:
		print("Failed safety test")
	print()
	print("Primary objective (log loss) evaluated on safety dataset:")
	print(SA.evaluate_primary_objective(branch='safety_test',theta=solution))
		
