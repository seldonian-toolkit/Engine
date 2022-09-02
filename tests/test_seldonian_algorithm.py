import pytest
import importlib
import autograd.numpy as np
import pandas as pd

from seldonian.utils.io_utils import load_json, load_pickle
from seldonian.utils.tutorial_utils import make_synthetic_regression_dataset
from seldonian.parse_tree.parse_tree import ParseTree,make_parse_trees_from_constraints
from seldonian.dataset import (DataSetLoader,
	SupervisedDataSet,RLDataSet)

from seldonian.spec import RLSpec, SupervisedSpec
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.models.models import LinearRegressionModel
from seldonian.models import objectives

import matplotlib.pyplot as plt

### Begin tests

def test_base_node_bound_methods_updated(gpa_regression_dataset):
	rseed=99
	np.random.seed(rseed) 
	constraint_strs = ['Mean_Squared_Error - 5.0','2.0 - Mean_Squared_Error']
	deltas = [0.05,0.05]
	(dataset,model_class,
		primary_objective,parse_trees) = gpa_regression_dataset(
			constraint_strs=constraint_strs,
			deltas=deltas)
	assert parse_trees[0].base_node_dict['Mean_Squared_Error']['bound_method'] == 'ttest'
	assert parse_trees[1].base_node_dict['Mean_Squared_Error']['bound_method'] == 'ttest'
	base_node_bound_method_dict = {
		'Mean_Squared_Error - 5.0': {
			'Mean_Squared_Error':'manual'
			},
		'2.0 - Mean_Squared_Error': {
			'Mean_Squared_Error':'random'
			}
	}
	frac_data_in_safety=0.6

	spec = SupervisedSpec(
		dataset=dataset,
		model_class=model_class,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=False,
		base_node_bound_method_dict=base_node_bound_method_dict,
		initial_solution_fn=model_class().fit,
		optimization_technique='barrier_function',
		optimizer='Powell',
		optimization_hyperparams={
			'maxiter'   : 1000,
			'seed':rseed,
			'hyper_search'  : None,
			'verbose'       : True,
		},
	)

	# Build SA object and verify that the bound method was updated
	SA = SeldonianAlgorithm(spec)
	assert parse_trees[0].base_node_dict['Mean_Squared_Error']['bound_method'] == 'manual'
	assert parse_trees[1].base_node_dict['Mean_Squared_Error']['bound_method'] == 'random'

def test_not_enough_data(generate_data):
	# dummy data for linear regression
	rseed=0
	np.random.seed(rseed) 
	numPoints=3
	columns=['feature1','label']
	model_class = LinearRegressionModel
	X,Y = generate_data(
		numPoints,loc_X=0.0,loc_Y=0.0,sigma_X=1.0,sigma_Y=1.0)
	rows = np.hstack([np.expand_dims(X,axis=1),np.expand_dims(Y,axis=1)])
	df = pd.DataFrame(rows,columns=columns)

	constraint_strs = ['Mean_Squared_Error - 2.0']
	deltas = [0.5]
	parse_trees = []
	
	for ii in range(len(constraint_strs)):
		constraint_str = constraint_strs[ii]

		delta = deltas[ii]
		# Create parse tree object
		parse_tree = ParseTree(delta=delta,
			regime='supervised_learning',sub_regime='regression',
			columns=[])

		# Fill out tree
		parse_tree.create_from_ast(constraint_str)
		# assign deltas for each base node
		# use equal weighting for each base node
		parse_tree.assign_deltas(weight_method='equal')

		# Assign bounds needed on the base nodes
		parse_tree.assign_bounds_needed()
		
		parse_trees.append(parse_tree)

	deltas = [0.05]

	dataset = SupervisedDataSet(df=df,meta_information=columns,
		label_column='label',
		sensitive_column_names=[],
		include_sensitive_columns=False,
		include_intercept_term=False
	)
	frac_data_in_safety=0.6

	# Create spec object
	# Will warn because of initial solution trying to fit with not enough data
	spec = SupervisedSpec(
		dataset=dataset,
		model_class=model_class,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=objectives.Mean_Squared_Error,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model_class().fit,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.01,
			'alpha_lamb'    : 0.01,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 200,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)
	warning_msg = "Warning: not enough data to run the Seldonian algorithm." 
	with pytest.warns(UserWarning,match=warning_msg) as excinfo:
		SA = SeldonianAlgorithm(spec)
		passed_safety,solution = SA.run()

	spec_zeros = SupervisedSpec(
		dataset=dataset,
		model_class=model_class,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=objectives.Mean_Squared_Error,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=lambda x,y: np.zeros(1),
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.01,
			'alpha_lamb'    : 0.01,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 200,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)
	warning_msg = "Warning: not enough data to run the Seldonian algorithm." 
	with pytest.warns(UserWarning,match=warning_msg) as excinfo:
		SA = SeldonianAlgorithm(spec_zeros)
		passed_safety,solution = SA.run()

def test_bad_optimizer(gpa_regression_dataset):
	""" Test that attempting to use an optimizer 
	or optimization_technique that is not supported
	raises an error """

	rseed=99
	np.random.seed(rseed) 
	constraint_strs = ['Mean_Squared_Error - 2.0']
	deltas = [0.05]
	(dataset,model_class,
		primary_objective,parse_trees) = gpa_regression_dataset(
			constraint_strs=constraint_strs,
			deltas=deltas)

	frac_data_in_safety=0.6

	bad_optimizer = 'bad-optimizer' 
	for optimization_technique in ['barrier_function','gradient_descent']:

		bad_spec = SupervisedSpec(
				dataset=dataset,
				model_class=model_class,
				parse_trees=parse_trees,
				sub_regime='regression',
				frac_data_in_safety=frac_data_in_safety,
				primary_objective=primary_objective,
				use_builtin_primary_gradient_fn=False,
				initial_solution_fn=model_class().fit,
				optimization_technique=optimization_technique,
				optimizer=bad_optimizer,
				optimization_hyperparams={
					'maxiter'   : 1000,
					'seed':rseed,
					'hyper_search'  : None,
					'verbose'       : True,
				},
			)

		# Run seldonian algorithm
		with pytest.raises(NotImplementedError) as excinfo:
			SA = SeldonianAlgorithm(bad_spec)
			passed_safety,solution = SA.run()
		error_str = "Optimizer: bad-optimizer is not supported"
		assert error_str in str(excinfo.value)

	bad_optimization_technique = 'bad-opt-technique' 

	bad_spec = SupervisedSpec(
			dataset=dataset,
			model_class=model_class,
			parse_trees=parse_trees,
			sub_regime='regression',
			frac_data_in_safety=frac_data_in_safety,
			primary_objective=primary_objective,
			use_builtin_primary_gradient_fn=False,
			initial_solution_fn=model_class().fit,
			optimization_technique=bad_optimization_technique,
			optimizer='adam',
			optimization_hyperparams={
				'maxiter'   : 1000,
				'seed':rseed,
				'hyper_search'  : None,
				'verbose'       : True,
			},
		)

	# Run seldonian algorithm
	with pytest.raises(NotImplementedError) as excinfo:
		SA = SeldonianAlgorithm(bad_spec)
		passed_safety,solution = SA.run()
	error_str = "Optimization technique: bad-opt-technique is not implemented"
	assert error_str in str(excinfo.value)

def test_phil_custom_base_node(gpa_regression_dataset):
	""" Test that the gpa regression example runs 
	using Phil's custom base node. Make
	sure safety test passes and solution is correct.
	"""
	rseed=0
	np.random.seed(rseed) 
	# constraint_strs = ['Mean_Squared_Error - 2.0']
	constraint_strs = ['MED_MF - 0.1']
	deltas = [0.05]

	(dataset,model_class,
		primary_objective,parse_trees) = gpa_regression_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	frac_data_in_safety=0.6

	# Create spec object
	spec = SupervisedSpec(
		dataset=dataset,
		model_class=model_class,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model_class().fit,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.01,
			'alpha_lamb'    : 0.01,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 10,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)

	# Run seldonian algorithm
	SA = SeldonianAlgorithm(spec)
	passed_safety,solution = SA.run(debug=True)
	assert passed_safety == True
	print(solution)
	array_to_compare = np.array(
		[ 0.42523186, -0.00285192, -0.00202239,
		 -0.00241261, -0.00234646, -0.0025831,
		  0.01924249,  0.01865552, -0.00308212, 
		 -0.0024446 ])

	assert np.allclose(solution,array_to_compare)

def test_cvar_custom_base_node():
	""" Test that the gpa regression example runs 
	using the custom base node that calculates 
	CVaR alpha of the squared error. Make
	sure safety test passes and solution is correct.

	Check that the actual value of the constraint (not the bound)
	is also correctly calculated.
	"""
	from seldonian.models.models import BoundedLinearRegressionModel
	rseed=0
	np.random.seed(rseed) 
	constraint_strs = ['CVaRSQE <= 50.0']
	deltas = [0.1]

	numPoints = 2500
	dataset = make_synthetic_regression_dataset(
		numPoints,
		loc_X=0.0,
		loc_Y=0.0,
		sigma_X=1.0,
		sigma_Y=0.2,
		include_intercept_term=False,clipped=True)

	parse_trees = make_parse_trees_from_constraints(
		constraint_strs,
		deltas)

	model_class = BoundedLinearRegressionModel

	# Create spec object
	spec = SupervisedSpec(
		dataset=dataset,
		model_class=model_class,
		sub_regime='regression',
		primary_objective=objectives.Mean_Squared_Error,
		use_builtin_primary_gradient_fn=False,
		custom_primary_gradient_fn=objectives.gradient_Bounded_Squared_Error,
		parse_trees=parse_trees,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.01,
			'alpha_lamb'    : 0.01,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 5,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)

	# Run seldonian algorithm
	SA = SeldonianAlgorithm(spec)
	passed_safety,solution = SA.run(debug=True)
	assert passed_safety == True
	solution_to_compare = np.array([[0.07197478]])
	assert np.allclose(solution,solution_to_compare)

def test_gpa_data_regression_multiple_constraints(gpa_regression_dataset):
	""" Test that the gpa regression example runs 
	with a two constraints using gradient descent. Make
	sure safety test passes and solution is correct.
	"""
	# Load metadata
	rseed=0
	np.random.seed(rseed) 
	constraint_strs = ['Mean_Squared_Error - 5.0','2.0 - Mean_Squared_Error']
	deltas = [0.05,0.1]

	(dataset,model_class,
		primary_objective,parse_trees) = gpa_regression_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	frac_data_in_safety=0.6

	# Create spec object
	spec = SupervisedSpec(
		dataset=dataset,
		model_class=model_class,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model_class().fit,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : np.array([0.5,0.5]),
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 200,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)

	# Run seldonian algorithm
	SA = SeldonianAlgorithm(spec)
	passed_safety,solution = SA.run()
	assert passed_safety == True
	array_to_compare = np.array(
		[ 4.18121191e-01,  7.65218366e-05,  8.68827231e-04,  4.96795941e-04,
		  5.40624536e-04,  3.35472715e-04,  2.10383120e-03,  1.52231771e-03,
		 -1.46634476e-04,  4.67094023e-04]
	)
	assert np.allclose(solution,array_to_compare)

def test_gpa_data_regression_custom_constraint(gpa_regression_dataset):
	""" Test that the gpa regression example runs 
	using Phil's custom base node: MED_MF. Make
	sure safety test passes and solution is correct.
	"""
	rseed=0
	np.random.seed(rseed) 
	constraint_strs = ['MED_MF - 0.2']
	deltas = [0.05]

	(dataset,model_class,
		primary_objective,parse_trees) = gpa_regression_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	frac_data_in_safety=0.6

	# Create spec object
	spec = SupervisedSpec(
		dataset=dataset,
		model_class=model_class,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model_class().fit,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 200,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)

	# Run seldonian algorithm
	SA = SeldonianAlgorithm(spec)
	passed_safety,solution = SA.run()
	assert passed_safety == True
	array_to_compare = np.array(
		[ 0.42155706, -0.00152678, -0.0006972,  -0.00108743, -0.00102126, -0.00125793,
  0.01056768,  0.00998072, -0.001757,   -0.00111942])

	assert np.allclose(solution,array_to_compare)

def test_gpa_data_classification(gpa_classification_dataset):
	""" Test that the gpa classification example runs 
	with the five fairness constraints (separately):
	Disparate impact
	Demographic parity
	Equalized odds
	Equal opportunity
	Predictive equality
	
	Make sure safety test passes and solution is correct.
	"""
	rseed=0
	np.random.seed(rseed)
	frac_data_in_safety=0.6

	fairness_constraint_dict = {
		'disparate_impact':'0.8 - min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M]))',
		'demographic_parity':'abs((PR | [M]) - (PR | [F])) - 0.15',
		'equalized_odds':'abs((FNR | [M]) - (FNR | [F])) + abs((FPR | [M]) - (FPR | [F])) - 0.35',
		'equal_opportunity':'abs((FNR | [M]) - (FNR | [F])) - 0.2',
		'predictive_equality':'abs((FPR | [M]) - (FPR | [F])) - 0.2'
		}

	solution_dict = {
	'disparate_impact':np.array(
		[-0.14932756, -0.04743285,  0.15603878,  0.10953721,  0.08014052,  0.03997749,
  0.40484586,  0.3045744,  -0.1084586,  -0.05770913]),
	'demographic_parity':np.array(
		[-0.14932756, -0.04743285,  0.15603878,  0.10953721,  0.08014052,  0.03997749,
  0.40484586,  0.3045744,  -0.1084586,  -0.05770913]),
	'equalized_odds':np.array(
		[-0.14932756, -0.04743285,  0.15603878,  0.10953721,  0.08014052,  0.03997749,
  0.40484586,  0.3045744,  -0.1084586,  -0.05770913]),
	'equal_opportunity':np.array(
		[-0.14932756, -0.04743285,  0.15603878,  0.10953721,  0.08014052,  0.03997749,
  0.40484586,  0.3045744,  -0.1084586,  -0.05770913]),
	'predictive_equality':np.array(
		[-0.14932756, -0.04743285,  0.15603878,  0.10953721,  0.08014052,  0.03997749,
  0.40484586,  0.3045744,  -0.1084586,  -0.05770913])
	}
	
	for constraint in fairness_constraint_dict:
		print(constraint)
		constraint_str = fairness_constraint_dict[constraint]
		constraint_strs = [constraint_str]
		deltas = [0.05]

		(dataset,model_class,
			primary_objective,parse_trees) = gpa_classification_dataset(
			constraint_strs=constraint_strs,
			deltas=deltas)

		# Create spec object
		spec = SupervisedSpec(
			dataset=dataset,
			model_class=model_class,
			parse_trees=parse_trees,
			sub_regime='classification',
			frac_data_in_safety=frac_data_in_safety,
			primary_objective=primary_objective,
			use_builtin_primary_gradient_fn=False,
			initial_solution_fn=model_class().fit,
			optimization_technique='gradient_descent',
			optimizer='adam',
			optimization_hyperparams={
				'lambda_init'   : np.array([0.5]),
				'alpha_theta'   : 0.005,
				'alpha_lamb'    : 0.005,
				'beta_velocity' : 0.9,
				'beta_rmsprop'  : 0.95,
				'num_iters'     : 10,
				'gradient_library': "autograd",
				'hyper_search'  : None,
				'verbose'       : True,
			}
		)

		# Run seldonian algorithm
		SA = SeldonianAlgorithm(spec)
		passed_safety,solution = SA.run()
		assert passed_safety == True
		print(solution)

		solution_to_compare = solution_dict[constraint]

		assert np.allclose(solution,solution_to_compare)

def test_classification_statistics(gpa_classification_dataset):
	""" Test all of the classification statistics (FPR, PR, NR, etc.)
	are evaluated properly for the GPA dataset
	where we know what the answers should be
	
	"""
	rseed=0
	np.random.seed(rseed)
	frac_data_in_safety=0.6
   
	constraint_str = '(PR + NR + FPR + FNR + TPR + TNR + logistic_loss) - 10.0'
	constraint_strs = [constraint_str]
	deltas = [0.05]

	(dataset,model_class,
		primary_objective,parse_trees) = gpa_classification_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	# Create spec object
	spec = SupervisedSpec(
		dataset=dataset,
		model_class=model_class,
		parse_trees=parse_trees,
		sub_regime='classification',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model_class().fit,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 25,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)

	# Run seldonian algorithm
	SA = SeldonianAlgorithm(spec)
	passed_safety,solution = SA.run()
	assert passed_safety == True
	print(passed_safety,solution)
	solution_to_compare = np.array(
		[-0.14932756, -0.04743285,  0.15603878,
		  0.10953721,  0.08014052,  0.03997749,
		  0.40484586,  0.3045744,  -0.1084586,  -0.05770913]
	)

	assert np.allclose(solution,solution_to_compare)

def test_NSF(gpa_regression_dataset):
	""" Test that no solution is found for a constraint
	that is impossible to satisfy, e.g. negative mean squared error 
	"""
	rseed=0
	np.random.seed(rseed) 
	constraint_strs = ['Mean_Squared_Error + 2.0'] 
	deltas = [0.05]

	(dataset,model_class,
		primary_objective,parse_trees) = gpa_regression_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	frac_data_in_safety=0.6

	# Create spec object
	spec = SupervisedSpec(
		dataset=dataset,
		model_class=model_class,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model_class().fit,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 200,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		},
	)

	# Run seldonian algorithm
	SA = SeldonianAlgorithm(spec)
	passed_safety,solution = SA.run()
	assert passed_safety == False
	assert solution == 'NSF'

def test_cmaes(gpa_regression_dataset):
	""" Test that the CMA-ES black box optimizers successfully optimize the GPA 
	regression problem with a simple non-conflicting constraint
	"""
	rseed=99
	np.random.seed(rseed) 
	constraint_strs = ['Mean_Squared_Error - 2.0']
	deltas = [0.05]
	(dataset,model_class,
		primary_objective,parse_trees) = gpa_regression_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	frac_data_in_safety=0.6

	array_to_compare = np.array(
		[4.17882264e-01, -1.59868384e-04,
		 6.33766780e-04,  2.64271363e-04,
		 3.08303718e-04,  1.01170148e-04,
		 1.86987938e-03,  1.29098726e-03,
		 -3.82405534e-04,  2.29938169e-04])

	# for optimizer in ['Powell','CG','Nelder-Mead','BFGS','CMA-ES']:
	for optimizer in ['CMA-ES']:
		spec = SupervisedSpec(
			dataset=dataset,
			model_class=model_class,
			parse_trees=parse_trees,
			sub_regime='regression',
			frac_data_in_safety=frac_data_in_safety,
			primary_objective=primary_objective,
			use_builtin_primary_gradient_fn=False,
			initial_solution_fn=model_class().fit,
			optimization_technique='barrier_function',
			optimizer=optimizer,
			optimization_hyperparams={
				'maxiter'   : 100 if optimizer == 'CMA-ES' else 1000,
				'seed':rseed,
				'hyper_search'  : None,
				'verbose'       : True,
			},
		)


		# Run seldonian algorithm
		SA = SeldonianAlgorithm(spec)
		passed_safety,solution = SA.run()

		assert passed_safety == True
		if optimizer != 'CMA-ES':
			# CMA-ES might come up with a different solution on test server
			assert np.allclose(solution,array_to_compare)

def test_use_custom_primary_gradient(gpa_regression_dataset):
	""" Test that the gpa regression example runs 
	when using a custom primary gradient function.
	It is the same as the built-in but passed as 
	a custom function. Make
	sure safety test passes and solution is correct.
	"""

	def gradient_MSE(model,theta,X,Y):
		n = len(X)
		prediction = model.predict(theta,X) # vector of values
		err = prediction-Y
		return 2/n*np.dot(err,X)

	rseed=0
	np.random.seed(rseed) 
	constraint_strs = ['Mean_Squared_Error - 2.0'] 
	deltas = [0.05]

	(dataset,model_class,
		primary_objective,parse_trees) = gpa_regression_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	frac_data_in_safety=0.6

	# Create spec object
	spec = SupervisedSpec(
		dataset=dataset,
		model_class=model_class,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=False,
		custom_primary_gradient_fn=gradient_MSE,
		initial_solution_fn=model_class().fit,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 200,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)

	# Run seldonian algorithm
	SA = SeldonianAlgorithm(spec)
	passed_safety,solution = SA.run()
	assert passed_safety == True
	array_to_compare = np.array(
		[ 4.17882259e-01, -1.59868384e-04,  6.33766780e-04,  2.64271363e-04,
  3.08303718e-04,  1.01170148e-04,  1.86987938e-03,  1.29098727e-03,
 -3.82405534e-04,  2.29938169e-04])
	assert np.allclose(solution,array_to_compare)

def test_get_candidate_selection_result(gpa_regression_dataset):
	""" Test that the after running the SA on the 
	gpa regression example, we can get the 
	full candidate selection solution dictionary
	from gradient descent as a method call on the
	SA() object.
	
	Also check that before we run SA.run() this same 
	method gives us an error. 
	"""

	rseed=0
	np.random.seed(rseed) 
	constraint_strs = ['Mean_Squared_Error - 2.0'] 
	deltas = [0.05]

	(dataset,model_class,
		primary_objective,parse_trees) = gpa_regression_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	frac_data_in_safety=0.6

	# Create spec object
	spec = SupervisedSpec(
		dataset=dataset,
		model_class=model_class,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model_class().fit,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 100,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)


	# # Run seldonian algorithm
	SA = SeldonianAlgorithm(spec)
	# Try to get candidate solution result before running
	with pytest.raises(ValueError) as excinfo:
		res = SA.get_cs_result()
	error_str = "Candidate selection has not been run yet, so result is not available.  Call run() first"
	assert error_str in str(excinfo.value)
	
	passed_safety,solution = SA.run(store_cs_values=True)
	res = SA.get_cs_result()
	res_keys = res.keys()
	for key in ['candidate_solution', 'best_index', 'best_feasible_g', 'best_feasible_f', 'solution_found', 'theta_vals', 'f_vals', 'g_vals', 'lamb_vals', 'L_vals']:
		assert key in res_keys


""" RL based tests """

# def test_RL_builtin_or_custom_gradient_not_supported():
# 	""" Test that an error is raised if user tries to 
# 	use built-in gradient or a custom gradient 
# 	when doing RL
# 	"""
# 	# Load data and metadata
# 	np.random.seed(0) 
# 	data_pth = 'static/datasets/RL/gridworld/gridworld3x3_50episodes.csv'
# 	metadata_pth = 'static/datasets/RL/gridworld/gridworld3x3_metadata.json'

# 	metadata_dict = load_json(metadata_pth)
# 	regime = metadata_dict['regime']
# 	columns = metadata_dict['columns']
				
# 	include_sensitive_columns = False
# 	include_intercept_term = False
# 	frac_data_in_safety = 0.6

# 	# Model
# 	model_class = TabularSoftmaxModel

# 	# RL environment
# 	RL_environment_name = metadata_dict['RL_environment_name']
# 	RL_environment_module = importlib.import_module(
# 		f'seldonian.RL.environments.{RL_environment_name}')
# 	RL_environment_obj = RL_environment_module.Environment()    

# 	# Primary objective
# 	model_instance = model_class(RL_environment_obj)

# 	primary_objective = model_instance.default_objective
# 	# Load dataset from file
# 	loader = DataSetLoader(
# 		regime=regime)

# 	dataset = loader.load_RL_dataset_from_csv(
# 		filename=data_pth,
# 		metadata_filename=metadata_pth)
	
# 	constraint_strs = ['-0.25 - J_pi_new'] 
	
# 	deltas = [0.05]

# 	# For each constraint, make a parse tree
# 	parse_trees = []
# 	for ii in range(len(constraint_strs)):
# 		constraint_str = constraint_strs[ii]

# 		delta = deltas[ii]
# 		# Create parse tree object
# 		parse_tree = ParseTree(delta=delta,regime='reinforcement_learning',
# 		sub_regime='all')

# 		# Fill out tree
# 		parse_tree.create_from_ast(constraint_str)
# 		# assign deltas for each base node
# 		# use equal weighting for each base node
# 		parse_tree.assign_deltas(weight_method='equal')

# 		# Assign bounds needed on the base nodes
# 		parse_tree.assign_bounds_needed()
		
# 		parse_trees.append(parse_tree)

# 	# # Create spec object
# 	spec = RLSpec(
# 		dataset=dataset,
# 		model_class=model_class,
# 		frac_data_in_safety=0.8,
# 		use_builtin_primary_gradient_fn=True,
# 		primary_objective=primary_objective,
# 		parse_trees=parse_trees,
# 		RL_environment_obj=RL_environment_obj,
# 		initial_solution_fn=None,
# 		bound_method='ttest',
# 		optimization_technique='gradient_descent',
# 		optimizer='adam',
# 		optimization_hyperparams={
# 			'lambda_init'   : np.array([0.5]),
# 			'alpha_theta'   : 0.005,
# 			'alpha_lamb'    : 0.005,
# 			'beta_velocity' : 0.9,
# 			'beta_rmsprop'  : 0.95,
# 			'num_iters'     : 20,
# 			'gradient_library': "autograd",
# 			'hyper_search'  : None,
# 			'verbose'       : True,
# 		},
# 		regularization_hyperparams={'reg_coef':0.1},
# 		normalize_returns=False,
# 	)

# 	# Run seldonian algorithm, making sure we capture error
# 	error_str = ("Using a builtin primary objective gradient"
# 				" is not yet supported for regimes other"
# 				" than supervised learning")
# 	with pytest.raises(NotImplementedError) as excinfo:
# 		SA = SeldonianAlgorithm(spec)
# 		passed_safety,solution = SA.run()
		
# 	assert error_str in str(excinfo.value)

# 	# # Create spec object
# 	spec2 = RLSpec(
# 		dataset=dataset,
# 		model_class=model_class,
# 		frac_data_in_safety=0.8,
# 		use_builtin_primary_gradient_fn=False,
# 		custom_primary_gradient_fn=lambda x: x,
# 		primary_objective=primary_objective,
# 		parse_trees=parse_trees,
# 		RL_environment_obj=RL_environment_obj,
# 		initial_solution_fn=None,
# 		bound_method='ttest',
# 		optimization_technique='gradient_descent',
# 		optimizer='adam',
# 		optimization_hyperparams={
# 			'lambda_init'   : np.array([0.5]),
# 			'alpha_theta'   : 0.005,
# 			'alpha_lamb'    : 0.005,
# 			'beta_velocity' : 0.9,
# 			'beta_rmsprop'  : 0.95,
# 			'num_iters'     : 20,
# 			'gradient_library': "autograd",
# 			'hyper_search'  : None,
# 			'verbose'       : True,
# 		},
# 		regularization_hyperparams={'reg_coef':0.1},
# 		normalize_returns=False,
# 	)

# 	# Run seldonian algorithm, making sure we capture error
# 	error_str2 = ("Using a provided primary objective gradient"
# 				" is not yet supported for regimes other"
# 				" than supervised learning")
# 	with pytest.raises(NotImplementedError) as excinfo2:
# 		SA = SeldonianAlgorithm(spec2)
# 		passed_safety,solution = SA.run()
		
# 	assert error_str2 in str(excinfo2.value)
	
# def test_RL_gridworld_gradient_descent(RL_gridworld_dataset):
# 	""" Test that the RL gridworld example runs 
# 	with a simple performance improvement constraint. Make
# 	sure safety test passes and solution is correct.
# 	"""
# 	# Load data and metadata
# 	rseed=99
# 	np.random.seed(rseed)
# 	constraint_strs = ['-0.25 - J_pi_new']
# 	deltas = [0.05]
	
# 	(dataset,model_class,
# 		primary_objective,
# 		parse_trees,
# 		RL_environment_obj) = RL_gridworld_dataset(
# 			constraint_strs=constraint_strs,
# 			deltas=deltas)

# 	frac_data_in_safety = 0.6
# 	# Create spec object
# 	spec = RLSpec(
# 		dataset=dataset,
# 		model_class=model_class,
# 		frac_data_in_safety=frac_data_in_safety,
# 		use_builtin_primary_gradient_fn=False,
# 		primary_objective=primary_objective,
# 		parse_trees=parse_trees,
# 		RL_environment_obj=RL_environment_obj,
# 		initial_solution_fn=None,
# 		bound_method='ttest',
# 		optimization_technique='gradient_descent',
# 		optimizer='adam',
# 		optimization_hyperparams={
# 			'alpha_theta'   : 0.005,
# 			'alpha_lamb'    : 0.005,
# 			'beta_velocity' : 0.9,
# 			'beta_rmsprop'  : 0.95,
# 			'num_iters'     : 15,
# 			'gradient_library': "autograd",
# 			'hyper_search'  : None,
# 			'verbose'       : True,
# 		},
# 		normalize_returns=False,
# 	)

# 	# # Run seldonian algorithm
# 	SA = SeldonianAlgorithm(spec)
# 	passed_safety,solution = SA.run()
# 	assert passed_safety == True
# 	array_to_compare = np.array(
# 	   [ 0.14021112, -0.14216886, -0.13786284, -0.14577261,  0.14013514,  0.13784917,
#  -0.14094687, -0.13881718,  0.13240779,  0.1411048,  -0.13954501,  0.1277282,
#   0.14035668, -0.14190627,  0.05335083, -0.13960011,  0.14010328, -0.14035336,
#   0.13488608,  0.14067562, -0.14114499,  0.14104931, -0.14068246,  0.14011436,
#  -0.13970751,  0.14057111, -0.1423285,   0.14027992, -0.13829985, -0.1416236,
#  -0.13949036,  0.14042179])
# 	assert np.allclose(solution,array_to_compare)

# def test_RL_gridworld_black_box(RL_gridworld_dataset):
# 	""" Test that the RL gridworld example runs 
# 	with a simple performance improvement constraint. Make
# 	sure safety test passes and solution is correct.
# 	"""
# 	# Load data and metadata
# 	rseed=99
# 	np.random.seed(rseed)
# 	constraint_strs = ['-0.25 - J_pi_new']
# 	deltas = [0.05]
	
# 	(dataset,model_class,
# 		primary_objective,
# 		parse_trees,
# 		RL_environment_obj) = RL_gridworld_dataset(
# 			constraint_strs=constraint_strs,
# 			deltas=deltas)

# 	frac_data_in_safety = 0.6
# 	# Create spec object
# 	spec = RLSpec(
# 		dataset=dataset,
# 		model_class=model_class,
# 		frac_data_in_safety=frac_data_in_safety,
# 		use_builtin_primary_gradient_fn=False,
# 		primary_objective=primary_objective,
# 		parse_trees=parse_trees,
# 		RL_environment_obj=RL_environment_obj,
# 		initial_solution_fn=None,
# 		bound_method='ttest',
# 		optimization_technique='barrier_function',
# 		optimizer='CMA-ES',
# 		optimization_hyperparams={
# 				'maxiter'   : 10,
# 				'seed':rseed,
# 				'hyper_search'  : None,
# 				'verbose'       : True,
# 			},
# 		normalize_returns=False,
# 	)

# 	# # Run seldonian algorithm
# 	SA = SeldonianAlgorithm(spec)
# 	passed_safety,solution = SA.run()
# 	assert passed_safety == True
# 	array_to_compare = np.array(
# 	   [-0.01478694, -1.52299897,  0.60606259, -0.0475612,  -0.13658868,  0.33656384,
#  -0.53947718, -0.2725281,   0.44593005,  0.86750493,  0.79728041,  0.10156789,
#   0.2646691,  -0.66778523,  0.6754071,   0.29783526,  0.49140432, -0.54893211,
#  -0.6835978,   0.93589933, -0.97537624,  1.19449046, -0.724231,    0.40491889,
#  -0.6061301,  -0.22444457, -0.76379049,  0.51380723,  0.86515137,  0.2919332,
#  -0.31993747,  0.27008568])
# 	assert np.allclose(solution,array_to_compare)
