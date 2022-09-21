import os
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
from seldonian.RL.RL_model import RL_model

import matplotlib.pyplot as plt

### Begin tests

def test_base_node_bound_methods_updated(gpa_regression_dataset):
	rseed=99
	np.random.seed(rseed) 
	constraint_strs = ['Mean_Squared_Error - 5.0','2.0 - Mean_Squared_Error']
	deltas = [0.05,0.05]
	(dataset,model,
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
		model=model,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=False,
		base_node_bound_method_dict=base_node_bound_method_dict,
		initial_solution_fn=model.fit,
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
	model = LinearRegressionModel()
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
		model=model,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=objectives.Mean_Squared_Error,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model.fit,
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
		model=model,
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
	(dataset,model,
		primary_objective,parse_trees) = gpa_regression_dataset(
			constraint_strs=constraint_strs,
			deltas=deltas)

	frac_data_in_safety=0.6

	bad_optimizer = 'bad-optimizer' 
	for optimization_technique in ['barrier_function','gradient_descent']:

		bad_spec = SupervisedSpec(
				dataset=dataset,
				model=model,
				parse_trees=parse_trees,
				sub_regime='regression',
				frac_data_in_safety=frac_data_in_safety,
				primary_objective=primary_objective,
				use_builtin_primary_gradient_fn=False,
				initial_solution_fn=model.fit,
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
			model=model,
			parse_trees=parse_trees,
			sub_regime='regression',
			frac_data_in_safety=frac_data_in_safety,
			primary_objective=primary_objective,
			use_builtin_primary_gradient_fn=False,
			initial_solution_fn=model.fit,
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

	(dataset,model,
		primary_objective,parse_trees) = gpa_regression_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	frac_data_in_safety=0.6

	# Create spec object
	spec = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model.fit,
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

	model = BoundedLinearRegressionModel()

	# Create spec object
	spec = SupervisedSpec(
		dataset=dataset,
		model=model,
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

	# Make sure we can evaluate constraint as well
	pt = parse_trees[0]
	pt.evaluate_constraint(theta=solution,dataset=dataset,
		model=model,regime='supervised_learning',
		branch='safety_test')
	assert pt.root.value == pytest.approx(-47.35451)

def test_cvar_lower_bound():
	""" The normal constraint only uses 
	the CVAR upper bound because we want CVAR < some value. 
	Test that the lower bound also works

	Check that the actual value of the constraint (not the bound)
	is also correctly calculated.
	"""
	from seldonian.models.models import BoundedLinearRegressionModel
	rseed=0
	np.random.seed(rseed) 
	constraint_strs = ['CVaRSQE >= 5.0']
	deltas = [0.1]

	numPoints = 1000
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

	model = BoundedLinearRegressionModel()

	# Create spec object
	spec = SupervisedSpec(
		dataset=dataset,
		model=model,
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
	solution_to_compare = np.array([-0.15467687])
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

	(dataset,model,
		primary_objective,parse_trees) = gpa_regression_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	frac_data_in_safety=0.6

	# Create spec object
	spec = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model.fit,
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

	(dataset,model,
		primary_objective,parse_trees) = gpa_regression_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	frac_data_in_safety=0.6

	# Create spec object
	spec = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model.fit,
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

		(dataset,model,
			primary_objective,parse_trees) = gpa_classification_dataset(
			constraint_strs=constraint_strs,
			deltas=deltas)

		# Create spec object
		spec = SupervisedSpec(
			dataset=dataset,
			model=model,
			parse_trees=parse_trees,
			sub_regime='classification',
			frac_data_in_safety=frac_data_in_safety,
			primary_objective=primary_objective,
			use_builtin_primary_gradient_fn=False,
			initial_solution_fn=model.fit,
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

	(dataset,model,
		primary_objective,parse_trees) = gpa_classification_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	# Create spec object
	spec = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='classification',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model.fit,
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
	that is impossible to satisfy, e.g. negative mean squared error.
	Make sure that candidate selection did return a solution though
	"""
	rseed=0
	np.random.seed(rseed) 
	constraint_strs = ['Mean_Squared_Error + 2.0'] 
	deltas = [0.05]

	(dataset,model,
		primary_objective,parse_trees) = gpa_regression_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	frac_data_in_safety=0.6

	# Create spec object
	spec = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model.fit,
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
	print(f"Solution: {solution}")
	assert passed_safety == False
	assert solution == 'NSF'

	res = SA.get_cs_result()
	candidate_solution = res['candidate_solution']
	assert isinstance(candidate_solution,np.ndarray)

def test_cmaes(gpa_regression_dataset):
	""" Test that the CMA-ES black box optimizers successfully optimize the GPA 
	regression problem with a simple non-conflicting constraint
	"""
	rseed=99
	np.random.seed(rseed) 
	constraint_strs = ['Mean_Squared_Error - 2.0']
	deltas = [0.05]
	(dataset,model,
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
			model=model,
			parse_trees=parse_trees,
			sub_regime='regression',
			frac_data_in_safety=frac_data_in_safety,
			primary_objective=primary_objective,
			use_builtin_primary_gradient_fn=False,
			initial_solution_fn=model.fit,
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

	(dataset,model,
		primary_objective,parse_trees) = gpa_regression_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	frac_data_in_safety=0.6

	# Create spec object
	spec = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=False,
		custom_primary_gradient_fn=gradient_MSE,
		initial_solution_fn=model.fit,
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

	(dataset,model,
		primary_objective,parse_trees) = gpa_regression_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	frac_data_in_safety=0.6

	# Create spec object
	spec = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model.fit,
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
	
	passed_safety,solution = SA.run()
	res = SA.get_cs_result()
	res_keys = res.keys()
	for key in ['candidate_solution', 'best_index', 'best_g', 'best_f', 'theta_vals', 'f_vals', 'g_vals', 'lamb_vals', 'L_vals']:
		assert key in res_keys

def test_nans_infs_gradient_descent(gpa_regression_dataset):
	""" Test that if nans or infs appear in theta in gradient
	descent then the algorithm returns whatever the best solution 
	has been so far.
	"""
	rseed=0
	np.random.seed(rseed) 
	constraint_strs = ['Mean_Squared_Error - 2.0'] 
	deltas = [0.05]

	(dataset,model,
		primary_objective,parse_trees) = gpa_regression_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	frac_data_in_safety=0.6
	# first nans
	initial_solution_fn_nan = lambda x,y: np.nan*np.ones(10)
	# Create spec object
	spec_nan = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=initial_solution_fn_nan,
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
	SA_nan = SeldonianAlgorithm(spec_nan)
	passed_safety_nan,solution_nan = SA_nan.run(debug=True)
	assert passed_safety_nan == False
	assert solution_nan == 'NSF'

	# now infs
	initial_solution_fn_inf = lambda x,y: np.inf*np.ones(10)
	# Create spec object
	spec_inf = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=initial_solution_fn_inf,
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
	SA_inf = SeldonianAlgorithm(spec_inf)
	passed_safety_inf,solution_inf = SA_inf.run(debug=True)
	assert passed_safety_inf == False
	assert solution_inf == 'NSF'
	
def test_run_safety_test_only(gpa_regression_dataset):
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

	(dataset,model,
		primary_objective,parse_trees) = gpa_regression_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	frac_data_in_safety=0.6

	# Create spec object
	spec = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model.fit,
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
	test_solution = np.array(
		[ 4.17882259e-01, -1.59868384e-04,  6.33766780e-04,  2.64271363e-04,
  3.08303718e-04,  1.01170148e-04,  1.86987938e-03,  1.29098727e-03,
 -3.82405534e-04,  2.29938169e-04])
	passed_safety,solution = SA.run_safety_test(test_solution)
	assert passed_safety == True
	assert np.allclose(test_solution,solution)

def test_reg_coef(gpa_regression_dataset):
	""" Test that using a regularization coefficient 
	works
	"""

	rseed=0
	np.random.seed(rseed) 
	constraint_strs = ['Mean_Squared_Error - 2.0'] 
	deltas = [0.05]

	(dataset,model,
		primary_objective,parse_trees) = gpa_regression_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	frac_data_in_safety=0.6

	# First gradient descent
	# Create spec object
	spec_gs = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model.fit,
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
		},
		regularization_hyperparams={
			'reg_coef':0.5
		}
	)

	# # Run seldonian algorithm
	SA_gs = SeldonianAlgorithm(spec_gs)
	# Try to get candidate solution result before running
	test_solution_gs = np.array(
		[ 4.17882259e-01, -1.59868384e-04,  6.33766780e-04,  2.64271363e-04,
  3.08303718e-04,  1.01170148e-04,  1.86987938e-03,  1.29098727e-03,
 -3.82405534e-04,  2.29938169e-04])
	passed_safety,solution = SA_gs.run()
	assert passed_safety == True
	assert np.allclose(test_solution_gs,solution)
		

	spec_bb = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model.fit,
		optimization_technique='barrier_function',
		optimizer='Powell',
		regularization_hyperparams={
			'reg_coef':0.5
		}
	)

	# # Run seldonian algorithm
	SA_bb = SeldonianAlgorithm(spec_bb)
	# Try to get candidate solution result before running
	test_solution_bb = np.array(
		[ 1.26219949e-04,  3.59203006e-04,  9.26674215e-04,  4.18683641e-04,
  3.62709523e-04,  3.48171863e-05,  1.90106843e-03,  1.31441205e-03,
 -6.56374856e-04,  2.12829138e-04])
	passed_safety,solution = SA_bb.run(debug=True)
	assert passed_safety == True
	assert np.allclose(test_solution_bb,solution)

def test_create_logfile(gpa_regression_dataset):
	""" Test that using a regularization coefficient 
	works
	"""
	# Check how many logs there are before test:
	logfiles_before = os.listdir('./logs/')
	n_before = len(logfiles_before)
	rseed=0
	np.random.seed(rseed) 
	constraint_strs = ['Mean_Squared_Error - 2.0'] 
	deltas = [0.05]

	(dataset,model,
		primary_objective,parse_trees) = gpa_regression_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	frac_data_in_safety=0.6

	# First gradient descent
	# Create spec object
	spec_gs = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model.fit,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 2,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		},
	)

	# # Run seldonian algorithm
	SA_gs = SeldonianAlgorithm(spec_gs)
	# Try to get candidate solution result before running
	passed_safety,solution = SA_gs.run(write_cs_logfile=True)
	logfiles_after = os.listdir('./logs/')
	n_after = len(logfiles_after)
	assert n_after == n_before + 1

def test_bad_autodiff_method(gpa_classification_dataset):
	""" Test that using a regularization coefficient 
	works
	"""
	constraint_str = 'PR >= 0.9'
	constraint_strs = [constraint_str]
	deltas = [0.05]

	(dataset,model,
		primary_objective,parse_trees) = gpa_classification_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)
	frac_data_in_safety = 0.6
	# Create spec object
	spec = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='classification',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model.fit,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 25,
			'gradient_library': "superfast",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)

	# Run seldonian algorithm
	SA = SeldonianAlgorithm(spec)
	with pytest.raises(NotImplementedError) as excinfo:
		passed_safety,solution = SA.run()

	error_str = "gradient library: superfast not supported"

	assert str(excinfo.value) == error_str

def test_lambda_init(gpa_regression_dataset):
	""" Test that the gpa regression example runs 
	with a two constraints using gradient descent. Make
	sure safety test passes and solution is correct.
	"""
	# Load metadata
	rseed=0
	np.random.seed(rseed) 
	constraint_strs = ['Mean_Squared_Error - 5.0','2.0 - Mean_Squared_Error']
	deltas = [0.05,0.1]

	(dataset,model,
		primary_objective,parse_trees) = gpa_regression_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	frac_data_in_safety=0.6
	hyperparams1 = {
			'lambda_init'   : 0.5,
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 2,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	# Create spec object
	spec1 = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model.fit,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams=hyperparams1
	)

	# Run seldonian algorithm
	SA1 = SeldonianAlgorithm(spec1)
	passed_safety,solution = SA1.run()
	
	hyperparams2 = {
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 2,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	# Create spec object
	spec2 = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model.fit,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams=hyperparams2
	)

	# Run seldonian algorithm
	SA2 = SeldonianAlgorithm(spec2)
	passed_safety,solution = SA2.run()
		
	hyperparams3 = {
			'lambda_init'   : np.array([[0.5]]),
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 2,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	# Create spec object
	spec3 = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		initial_solution_fn=model.fit,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams=hyperparams3
	)

	# Run seldonian algorithm
	SA3 = SeldonianAlgorithm(spec3)
	with pytest.raises(RuntimeError) as excinfo:
		passed_safety,solution = SA3.run()

	error_str = "lambda has wrong shape. Shape must be (n_constraints,1)"
	assert str(excinfo.value) == error_str

def test_no_primary_provided(gpa_regression_dataset,
	gpa_classification_dataset,RL_gridworld_dataset):
	""" Test that if the user does not provide a primary objective,
	then the default is used in the three different regimes/sub-regimes
	"""
	# Regression
	rseed=99
	np.random.seed(rseed) 
	constraint_strs = ['Mean_Squared_Error - 2.0']
	deltas = [0.05]
	(dataset,model,
		_,parse_trees) = gpa_regression_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)
	frac_data_in_safety=0.6

	spec = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=None,
		use_builtin_primary_gradient_fn=False,
		initial_solution_fn=model.fit,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : 0.5,
			'alpha_theta'   : 0.01,
			'alpha_lamb'    : 0.01,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 2,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
		
	)
	assert spec.primary_objective == None

	# Create seldonian algorithm object, which assigns primary objective
	SA = SeldonianAlgorithm(spec)
	assert spec.primary_objective != None
	assert spec.primary_objective.__name__ == "Mean_Squared_Error"

	# Classification
	constraint_strs = ["FPR - 0.5"]
	deltas = [0.05]

	(dataset,model,
		_,parse_trees) = gpa_classification_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	# Create spec object
	spec = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='classification',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=None,
		use_builtin_primary_gradient_fn=False,
		initial_solution_fn=model.fit,
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
	assert spec.primary_objective == None

	# Create seldonian algorithm object, which assigns primary objective
	SA = SeldonianAlgorithm(spec)
	assert spec.primary_objective != None
	assert spec.primary_objective.__name__ == "logistic_loss"

	# RL 
	constraint_strs = ['-0.25 - J_pi_new']
	deltas = [0.05]
	
	parse_trees = make_parse_trees_from_constraints(
		constraint_strs,
	    deltas,
	    regime='reinforcement_learning',
	    sub_regime='all',
	    delta_weight_method='equal')
	(dataset,policy,
		env_kwargs,_) = RL_gridworld_dataset()
				
	frac_data_in_safety = 0.6

	# Model

	model = RL_model(policy=policy,env_kwargs=env_kwargs)

	# Create spec object
	spec = RLSpec(
		dataset=dataset,
		model=model,
		frac_data_in_safety=frac_data_in_safety,
		use_builtin_primary_gradient_fn=True,
		primary_objective=None,
		parse_trees=parse_trees,
		initial_solution_fn=None,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : 0.5,
			'alpha_theta'   : 0.01,
			'alpha_lamb'    : 0.01,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 2,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		},
	)
	assert spec.primary_objective == None

	# Create seldonian algorithm object, which assigns primary objective
	SA = SeldonianAlgorithm(spec)
	assert spec.primary_objective != None
	assert spec.primary_objective.__name__ == "IS_estimate"

""" RL based tests """

def test_RL_builtin_or_custom_gradient_not_supported(
	RL_gridworld_dataset):
	""" Test that an error is raised if user tries to 
	use built-in gradient or a custom gradient 
	when doing RL
	"""
	rseed=99
	np.random.seed(rseed)
	constraint_strs = ['-0.25 - J_pi_new']
	deltas = [0.05]
	
	parse_trees = make_parse_trees_from_constraints(
		constraint_strs,
	    deltas,
	    regime='reinforcement_learning',
	    sub_regime='all',
	    columns=[],
	    delta_weight_method='equal')
	(dataset,policy,
		env_kwargs,primary_objective) = RL_gridworld_dataset()
				
	frac_data_in_safety = 0.6

	# Model

	model = RL_model(policy=policy,env_kwargs=env_kwargs)

	# Create spec object
	spec = RLSpec(
		dataset=dataset,
		model=model,
		frac_data_in_safety=frac_data_in_safety,
		use_builtin_primary_gradient_fn=True,
		primary_objective=primary_objective,
		parse_trees=parse_trees,
		initial_solution_fn=None,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : 0.5,
			'alpha_theta'   : 0.01,
			'alpha_lamb'    : 0.01,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 2,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		},
	)

	# Run seldonian algorithm, making sure we capture error
	error_str = ("Using a builtin primary objective gradient"
				" is not yet supported for regimes other"
				" than supervised learning")
	with pytest.raises(NotImplementedError) as excinfo:
		SA = SeldonianAlgorithm(spec)
		passed_safety,solution = SA.run()
		
	assert error_str in str(excinfo.value)

	# # # Create spec object
	spec2 = RLSpec(
		dataset=dataset,
		model=model,
		frac_data_in_safety=frac_data_in_safety,
		use_builtin_primary_gradient_fn=False,
		custom_primary_gradient_fn=lambda x: x,
		primary_objective=primary_objective,
		parse_trees=parse_trees,
		initial_solution_fn=None,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : 0.5,
			'alpha_theta'   : 0.01,
			'alpha_lamb'    : 0.01,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 2,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		},
	)
	
	# Run seldonian algorithm, making sure we capture error
	error_str2 = ("Using a provided primary objective gradient"
				" is not yet supported for regimes other"
				" than supervised learning")
	with pytest.raises(NotImplementedError) as excinfo2:
		SA = SeldonianAlgorithm(spec2)
		passed_safety,solution = SA.run()
		
	assert error_str2 in str(excinfo2.value)
	
def test_RL_gridworld_gradient_descent(RL_gridworld_dataset):
	""" Test that the RL gridworld example runs 
	with a simple performance improvement constraint. Make
	sure safety test passes and solution is correct.
	"""
	# Load data and metadata
	rseed=99
	np.random.seed(rseed)
	constraint_strs = ['-10.0 - J_pi_new']
	deltas = [0.05]
	
	parse_trees = make_parse_trees_from_constraints(
		constraint_strs,
	    deltas,
	    regime='reinforcement_learning',
	    sub_regime='all',
	    columns=[],
	    delta_weight_method='equal')
	(dataset,policy,
		env_kwargs,primary_objective) = RL_gridworld_dataset()

	frac_data_in_safety = 0.6
	model = RL_model(policy=policy,env_kwargs=env_kwargs)
	# Create spec object
	spec = RLSpec(
		dataset=dataset,
		model=model,
		frac_data_in_safety=frac_data_in_safety,
		use_builtin_primary_gradient_fn=False,
		primary_objective=primary_objective,
		parse_trees=parse_trees,
		initial_solution_fn=None,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : 0.5,
			'alpha_theta'   : 0.01,
			'alpha_lamb'    : 0.01,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 5,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		},
	)

	# # Run seldonian algorithm
	SA = SeldonianAlgorithm(spec)
	passed_safety,solution = SA.run()
	assert passed_safety == True
	g_vals = SA.cs_result['g_vals']
	assert g_vals[1][0] == pytest.approx(-9.50428271)

	#Get primary objective
	primary_val = SA.evaluate_primary_objective(theta=solution,branch='safety_test')
	assert primary_val == pytest.approx(0.285780384)

def test_RL_gridworld_black_box(RL_gridworld_dataset):
	""" Test that trying to run RL example with 
	black box optimization gives a NotImplementedError,
	because it is not yet supported 
	"""
	# Load data and metadata
	rseed=99
	np.random.seed(rseed)
	constraint_strs = ['-0.25 - J_pi_new']
	deltas = [0.05]
	
	parse_trees = make_parse_trees_from_constraints(
		constraint_strs,
	    deltas,
	    regime='reinforcement_learning',
	    sub_regime='all',
	    columns=[],
	    delta_weight_method='equal')
	(dataset,policy,
		env_kwargs,primary_objective) = RL_gridworld_dataset()

	frac_data_in_safety = 0.6
	model = RL_model(policy=policy,env_kwargs=env_kwargs)
	# Create spec object
	spec = RLSpec(
		dataset=dataset,
		model=model,
		frac_data_in_safety=frac_data_in_safety,
		use_builtin_primary_gradient_fn=False,
		primary_objective=primary_objective,
		initial_solution_fn = None,
		parse_trees=parse_trees,
		optimization_technique='barrier_function',
		optimizer='Powell',
		optimization_hyperparams={
				'maxiter'   : 1000,
				'seed':rseed,
				'hyper_search'  : None,
				'verbose'       : True,
			},
	)

	# # Run seldonian algorithm
	with pytest.raises(NotImplementedError) as excinfo:
		SA = SeldonianAlgorithm(spec)
		passed_safety,solution = SA.run()
	error_str = (
				"barrier_function optimization_technique "
				"is not supported for reinforcement learning. "
				"Use gradient_descent instead.")

	assert error_str in str(excinfo.value)
