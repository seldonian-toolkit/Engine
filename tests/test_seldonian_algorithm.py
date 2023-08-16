import os
import pytest
import importlib
import autograd.numpy as np
import pandas as pd

from seldonian.utils.io_utils import load_json, load_pickle
from seldonian.utils.tutorial_utils import (
	make_synthetic_regression_dataset,generate_data)
from seldonian.parse_tree.parse_tree import ParseTree,make_parse_trees_from_constraints
from seldonian.dataset import (DataSetLoader,
	SupervisedDataSet,RLDataSet)

from seldonian.spec import (RLSpec, SupervisedSpec,
	createSupervisedSpec)
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.models.models import *
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

def test_not_enough_data(simulated_regression_dataset):
	# dummy data for linear regression

	constraint_strs = ['Mean_Squared_Error - 2.0']
	deltas = [0.5]
	numPoints=3
	(dataset,model,primary_objective,
        parse_trees) = simulated_regression_dataset(
            constraint_strs,deltas,numPoints=numPoints)
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
			'use_batches'   : False,
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
		initial_solution_fn=lambda x,y: np.zeros(2),
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.01,
			'alpha_lamb'    : 0.01,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 200,
			'use_batches'   : False,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)
	warning_msg = "Warning: not enough data to run the Seldonian algorithm." 
	with pytest.warns(UserWarning,match=warning_msg) as excinfo:
		SA = SeldonianAlgorithm(spec_zeros)
		passed_safety,solution = SA.run()

def test_data_as_lists(simulated_regression_dataset_aslists):
	# dummy data for linear regression

	constraint_strs = ['Mean_Squared_Error - 2.0']
	deltas = [0.5]
	numPoints=1000
	(dataset,model,primary_objective,
        parse_trees) = simulated_regression_dataset_aslists(
            constraint_strs,deltas,numPoints=numPoints)
	frac_data_in_safety=0.6

	# Create spec object
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
			'use_batches'   : False,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)
	
	SA = SeldonianAlgorithm(spec)
	candidate_features = SA.candidate_dataset.features
	candidate_labels = SA.candidate_dataset.labels
	assert type(candidate_features) == list
	assert type(candidate_labels) == np.ndarray

	with pytest.raises(NotImplementedError) as excinfo:
		SA = SeldonianAlgorithm(spec)
		passed_safety,solution = SA.run()
	error_str = (
		"This function is not supported when features are in a list. "
		"Convert features to a numpy array if possible or use autodiff "
		" to get the gradient.")
	assert str(excinfo.value) == error_str

	# Create spec object using autodiff
	spec2 = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=objectives.Mean_Squared_Error,
		use_builtin_primary_gradient_fn=False,
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
			'use_batches'   : False,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)
	
	SA2 = SeldonianAlgorithm(spec2)
	candidate_features = SA2.candidate_dataset.features
	candidate_labels = SA2.candidate_dataset.labels
	assert type(candidate_features) == list
	assert type(candidate_labels) == np.ndarray

	passed_safety,solution = SA2.run()
	assert passed_safety == True
	array_to_compare = np.array([0.02483889,0.98311923,0.02349485])
	assert np.allclose(solution,array_to_compare)

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
			'use_batches'   : False,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)

	# Run seldonian algorithm
	SA = SeldonianAlgorithm(spec)
	passed_safety,solution = SA.run(debug=True)
	assert passed_safety == True

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
		clipped=True)

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
			'use_batches'   : False,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)

	# Run seldonian algorithm
	SA = SeldonianAlgorithm(spec)
	passed_safety,solution = SA.run(debug=True)
	assert passed_safety == True
	solution_to_compare = np.array([-0.07257342,0.07182381])
	assert np.allclose(solution,solution_to_compare)

	# Make sure we can evaluate constraint as well
	pt = parse_trees[0]
	pt.evaluate_constraint(theta=solution,dataset=dataset,
		model=model,regime='supervised_learning',
		branch='safety_test')
	assert pt.root.value == pytest.approx(-47.163772762)

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
		clipped=True)

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
			'use_batches'   : False,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)

	# Run seldonian algorithm
	SA = SeldonianAlgorithm(spec)
	passed_safety,solution = SA.run()
	assert passed_safety == True
	solution_to_compare = np.array([-0.15426298, -0.15460036])
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
			'use_batches'   : False,
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
			'use_batches'   : False,
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
				'use_batches'   : False,
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
   
	constraint_str = '(PR + NR + FPR + FNR + TPR + TNR) - 10.0'
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
			'use_batches'   : False,
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
			'num_iters'     : 100,
			'use_batches'   : False,
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

	# Test that evaluate primary objective function raises a value error
	with pytest.raises(ValueError) as excinfo:
		SA.evaluate_primary_objective(
			branch="candidate_solution",theta=solution)

	assert str(excinfo.value) == "Cannot evaluate primary objective because theta='NSF'"

	# Test that evaluate primary objective function raises a value error
	with pytest.raises(ValueError) as excinfo:
		SA.evaluate_primary_objective(
			branch="safety_test",theta=solution)

	assert str(excinfo.value) == "Cannot evaluate primary objective because theta='NSF'"

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
		custom_primary_gradient_fn=objectives.gradient_Mean_Squared_Error,
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
			'use_batches'   : False,
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
			'use_batches'   : False,
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
	error_str = "Candidate selection has not been run yet, so result is not available."
	assert error_str in str(excinfo.value)
	
	passed_safety,solution = SA.run()
	res = SA.get_cs_result()
	res_keys = res.keys()
	for key in ['candidate_solution', 'best_index', 'best_g', 'best_f', 'f_vals', 'g_vals', 'lamb_vals', 'L_vals']:
		assert key in res_keys

def test_get_safety_test_result(gpa_regression_dataset):
	""" Test that the after running the SA on the 
	gpa regression example, we can get the 
	dictionary containing the parse trees evaluated 
	on the safety test. We also test the method
	that retrieves the upper bounds.
	
	Also check that before we run SA.run() this same 
	method gives us an error. 
	"""

	rseed=0
	np.random.seed(rseed) 
	constraint_strs = ['Mean_Squared_Error >= 1.25',
        'Mean_Squared_Error <= 2.0']
	deltas = [0.1,0.1]

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
			'lambda_init'   : 0.5,
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 150,
			'use_batches'   : False,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)


	# # Run seldonian algorithm
	SA = SeldonianAlgorithm(spec)
	# Try to get candidate solution result before running
	with pytest.raises(ValueError) as excinfo:
		res = SA.get_st_upper_bounds()
	error_str = "Safety test has not been run yet, so upper bounds are not available."
	assert error_str in str(excinfo.value)
	
	passed_safety,solution = SA.run()
	assert passed_safety == True
	res = SA.get_st_upper_bounds()
	assert len(res) == 2
	print(res)
	assert res['1.25-(Mean_Squared_Error)'] == pytest.approx(-0.19604227384297923)
	assert res['Mean_Squared_Error-(2.0)'] == pytest.approx(-0.5219448029759275)

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
			'use_batches'   : False,
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
			'use_batches'   : False,
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
			'use_batches'   : False,
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
			'use_batches'   : False,
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
	log_dir = "./logs/"
	os.makedirs(log_dir,exist_ok=True)
	logfiles_before = os.listdir(log_dir)
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
			'use_batches'   : False,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		},
	)

	# # Run seldonian algorithm
	SA_gs = SeldonianAlgorithm(spec_gs)
	# Try to get candidate solution result before running
	passed_safety,solution = SA_gs.run(write_cs_logfile=True)
	logfiles_after = os.listdir(log_dir)
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
			'use_batches'   : False,
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
	""" Test that lambda given with correct shape
	works but with wrong shape raises an error
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

	# A float can work - assumption is that all constraints get this value
	hyperparams1 = {
			'lambda_init'   : 0.5,
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 2,
			'use_batches'   : False,
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
			'lambda_init'   : np.array([0.5,0.25]),
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 2,
			'use_batches'   : False,
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
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 2,
			'use_batches'   : False,
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

	error_str = ("lambda has wrong shape. "
		"Shape must be (n_constraints,), "
		"but shape is (1,)")
	assert str(excinfo.value) == error_str

	# Allow a list to be passed
	hyperparams4 = {
			'lambda_init'   : [0.05,0.15],
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 2,
			'use_batches'   : False,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	# Create spec object
	spec4 = SupervisedSpec(
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
		optimization_hyperparams=hyperparams4
	)

	# Run seldonian algorithm
	SA4 = SeldonianAlgorithm(spec4)
	
	passed_safety,solution = SA4.run()

	# But not a list of the wrong length
	hyperparams5 = {
			'lambda_init'   : [0.05],
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 2,
			'use_batches'   : False,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	# Create spec object
	spec5 = SupervisedSpec(
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
		optimization_hyperparams=hyperparams5
	)

	# Run seldonian algorithm
	SA5 = SeldonianAlgorithm(spec5)
	

	with pytest.raises(RuntimeError) as excinfo:
		passed_safety,solution = SA5.run()

	error_str = ("lambda has wrong shape. "
		"Shape must be (n_constraints,), "
		"but shape is (1,)")
	assert str(excinfo.value) == error_str

	# Don't allow an array that has too many dimensions
	hyperparams6 = {
			'lambda_init'   : np.array([[0.05]]),
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 2,
			'use_batches'   : False,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	# Create spec object
	spec6 = SupervisedSpec(
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
		optimization_hyperparams=hyperparams6
	)

	# Run seldonian algorithm
	SA6 = SeldonianAlgorithm(spec6)
	

	with pytest.raises(RuntimeError) as excinfo:
		passed_safety,solution = SA6.run()

	error_str = ("lambda has wrong shape. "
		"Shape must be (n_constraints,), "
		"but shape is (1, 1)")
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
			'use_batches'   : False,
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
			'use_batches'   : False,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)
	assert spec.primary_objective == None

	# Create seldonian algorithm object, which assigns primary objective
	SA = SeldonianAlgorithm(spec)
	assert spec.primary_objective != None
	assert spec.primary_objective.__name__ == "binary_logistic_loss"

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
			'use_batches'   : False,
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

def test_no_initial_solution_provided(gpa_regression_dataset,
	gpa_classification_dataset,gpa_multiclass_dataset,
	RL_gridworld_dataset):
	""" Test that if the user does not provide a primary objective,
	then the default is used in the three different regimes/sub-regimes
	"""
	# Regression
	rseed=99
	np.random.seed(rseed) 
	constraint_strs = ['Mean_Squared_Error - 2.0']
	deltas = [0.05]
	(dataset,model,
		primary_objective,parse_trees) = gpa_regression_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)
	frac_data_in_safety=0.6

	spec = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='regression',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=False,
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
			'use_batches'   : False,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
		
	)
	SA = SeldonianAlgorithm(spec)
	SA.set_initial_solution()
	assert np.allclose(SA.initial_solution,np.zeros(10))

	# Binary Classification
	constraint_strs = ["FPR - 0.5"]
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
		sub_regime='binary_classification',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=False,
		initial_solution_fn=None,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 2,
			'use_batches'   : False,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)

	# Create seldonian algorithm object
	SA = SeldonianAlgorithm(spec)
	SA.set_initial_solution()
	assert np.allclose(SA.initial_solution,np.zeros(10))
	
	# Multi-class Classification
	constraint_strs = ["CM_[0,0] >= 0.25"]
	deltas = [0.05]

	(dataset,model,
		primary_objective,parse_trees) = gpa_multiclass_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)

	# Create spec object

	spec = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		sub_regime='multiclass_classification',
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=False,
		initial_solution_fn=None,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 2,
			'use_batches'   : False,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)

	# Create seldonian algorithm object
	SA = SeldonianAlgorithm(spec)
	SA.set_initial_solution()
	assert np.allclose(SA.initial_solution,np.zeros((10,3)))
	
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
			'use_batches'   : False,
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
			'use_batches'   : False,
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
			'use_batches'   : False,
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
	assert g_vals[1][0] == pytest.approx(-9.67469087)

	#Get primary objective
	primary_val_cs = SA.evaluate_primary_objective(
		theta=solution,branch='candidate_selection')
	primary_val_st = SA.evaluate_primary_objective(theta=solution,branch='safety_test')
	assert primary_val_st == pytest.approx(0.42407173678433796)

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
		passed_safety,solution = SA.run(debug=True)
	error_str = (
				"Optimizer: Powell "
				"is not supported for reinforcement learning. "
				"Try optimizer='CMA-ES' instead.")

	assert error_str in str(excinfo.value)

def test_RL_gridworld_alt_rewards(RL_gridworld_dataset_alt_rewards):
	""" Test that we can put constraints on returns that use alternate rewards
	"""
	rseed=99
	np.random.seed(rseed)

	# Vanilla IS first
	IS_constraint_strs = ['-0.25 - J_pi_new_[1]']
	deltas = [0.05]
	
	IS_parse_trees = make_parse_trees_from_constraints(
		IS_constraint_strs,
		deltas,
		regime='reinforcement_learning',
		sub_regime='all',
		columns=[],
		delta_weight_method='equal')
	(dataset,policy,
		env_kwargs,primary_objective) = RL_gridworld_dataset_alt_rewards()

	frac_data_in_safety = 0.6
	model = RL_model(policy=policy,env_kwargs=env_kwargs)
	# Create spec object
	IS_spec = RLSpec(
		dataset=dataset,
		model=model,
		frac_data_in_safety=frac_data_in_safety,
		use_builtin_primary_gradient_fn=False,
		primary_objective=primary_objective,
		initial_solution_fn = None,
		parse_trees=IS_parse_trees,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : 0.5,
			'alpha_theta'   : 0.01,
			'alpha_lamb'    : 0.01,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 5,
			'use_batches'   : False,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		},
	)

	# # Run seldonian algorithm
	IS_SA = SeldonianAlgorithm(IS_spec)
	passed_safety,solution = IS_SA.run()

	## now PDIS
	PDIS_constraint_strs = ['-0.25 - J_pi_new_PDIS_[1]']
	deltas = [0.05]
	
	PDIS_parse_trees = make_parse_trees_from_constraints(
		PDIS_constraint_strs,
		deltas,
		regime='reinforcement_learning',
		sub_regime='all',
		columns=[],
		delta_weight_method='equal')

	PDIS_spec = RLSpec(
		dataset=dataset,
		model=model,
		frac_data_in_safety=frac_data_in_safety,
		use_builtin_primary_gradient_fn=False,
		primary_objective=primary_objective,
		initial_solution_fn = None,
		parse_trees=PDIS_parse_trees,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : 0.5,
			'alpha_theta'   : 0.01,
			'alpha_lamb'    : 0.01,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 5,
			'use_batches'   : False,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		},
	)

	# # Run seldonian algorithm
	PDIS_SA = SeldonianAlgorithm(PDIS_spec)
	passed_safety,solution = PDIS_SA.run()


