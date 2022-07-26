import os

from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.io_utils import load_pickle

if __name__ == '__main__':
	# loan spec file
	# interface_output_dir = os.path.join('/Users/ahoag/beri/code',
	# 	'interface_outputs/loan_demographic_parity')
	# specfile = os.path.join(interface_output_dir,'spec.pkl')
	specfile = './spec.pkl'
	spec = load_pickle(specfile)
	spec.primary_objective = spec.model_class().sample_logistic_loss
	
	spec.use_builtin_primary_gradient_fn = False
	spec.optimization_hyperparams['alpha_theta'] = 0.01
	spec.optimization_hyperparams['alpha_lamb'] = 0.01
	spec.optimization_hyperparams['num_iters'] = 1000
	SA = SeldonianAlgorithm(spec)
	passed_safety,solution = SA.run()
	print(passed_safety)
	# if passed_safety:
	# 	print()
	# 	print("Primary objective evaluated on safety test:")
	# 	print(SA.evaluate_primary_objective(branch='safety_test',theta=solution))
