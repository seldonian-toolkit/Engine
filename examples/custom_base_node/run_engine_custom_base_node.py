import autograd.numpy as np
from seldonian.utils.tutorial_utils import make_synthetic_regression_dataset
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.models.models import LinearRegressionModel
from seldonian.parse_tree.parse_tree import make_parse_trees_from_constraints
from seldonian.spec import SupervisedSpec
from seldonian.models.models import SquashedLinearRegressionModel

import matplotlib.pyplot as plt

def main():
	""" Test that the gpa regression example runs 
	using the custom base node that calculates 
	CVaR alpha of the squared error. Make
	sure safety test passes and solution is correct.

	Check that the actual value of the constraint (not the bound)
	is also correctly calculated.
	"""
	rseed=0
	np.random.seed(rseed) 
	constraint_strs = ['CVARSQE <= 10.0']
	# constraint_strs = ['Mean_Squared_Error <= 4.0']
	deltas = [0.05]

	numPoints = 100000
	dataset = make_synthetic_regression_dataset(numPoints,
		include_intercept_term=True,clipped=True)
	parse_trees = make_parse_trees_from_constraints(
		constraint_strs,
		deltas)
	def init_solution(*args,**kwargs):
		return np.array([-0.1,0.5])
	model_class = SquashedLinearRegressionModel
	# Create spec object
	# Will warn because of initial solution trying to fit with not enough data
	spec = SupervisedSpec(
		dataset=dataset,
		model_class=SquashedLinearRegressionModel,
		sub_regime='regression',
		primary_objective=model_class().sample_Mean_Squared_Error,
		use_builtin_primary_gradient_fn=True,
		parse_trees=parse_trees,
		# initial_solution_fn=init_solution,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.01,
			'alpha_lamb'    : 0.01,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 50,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)
	
	# Run seldonian algorithm
	SA = SeldonianAlgorithm(spec)
	passed_safety,solution = SA.run(write_cs_logfile=True)
	print(passed_safety,solution)
	# last_theta = np.array([-0.4431018])
	# candidate_features = np.array(SA.candidate_features)
	# candidate_labels = np.array(SA.candidate_labels)
	# candidate_predictions = spec.model_class().predict(
	# 	last_theta,candidate_features)

	# ax.scatter(candidate_labels,candidate_predictions)
	# ax.set_xlabel("y (candidate)")
	# ax.set_ylabel("y_hat (candidate)")
	# ax.set_title("Labels vs. predictions after candidate selection")
	# ax.set_xlim(-4,4)
	# ax.set_ylim(-4,4)
	plt.show()

if __name__ == "__main__":
	main()