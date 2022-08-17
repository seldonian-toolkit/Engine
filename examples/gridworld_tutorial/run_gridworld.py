from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.io_utils import load_pickle

if __name__ == '__main__':
	# load specfile
	specfile = 'spec.pkl'
	spec = load_pickle(specfile)
	spec.optimization_hyperparams['num_iters']=3
	# Run Seldonian algorithm 
	SA = SeldonianAlgorithm(spec)
	passed_safety,solution = SA.run()
	if passed_safety:
		print("Passed safety test")
		print("The solution found is:")
		print(solution)
	else:
		print("No Solution Found")