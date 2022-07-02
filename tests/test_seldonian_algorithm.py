import pytest
import importlib
import autograd.numpy as np
import pandas as pd

from seldonian.utils.io_utils import load_json, load_pickle
from seldonian.parse_tree.parse_tree import ParseTree
from seldonian.dataset import (DataSetLoader,
	SupervisedDataSet,RLDataSet)
from seldonian.models.model import (LinearRegressionModel,
	TabularSoftmaxModel)
from seldonian.spec import RLSpec, SupervisedSpec
from seldonian.seldonian_algorithm import seldonian_algorithm



### Begin tests

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
			regime='supervised',sub_regime='regression',
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
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=model_class().default_objective,
		use_builtin_primary_gradient_fn=True,
		parse_trees=parse_trees,
		initial_solution_fn=model_class().fit,
		bound_method='ttest',
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
	warning_msg = ("Warning: not enough data to run the algorithm.  " 
                   "Returning NSF and failing safety test.") 
	with pytest.warns(UserWarning,match=warning_msg) as excinfo:
		passed_safety,solution = seldonian_algorithm(spec)

	spec_zeros = SupervisedSpec(
		dataset=dataset,
		model_class=model_class,
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=model_class().default_objective,
		use_builtin_primary_gradient_fn=True,
		parse_trees=parse_trees,
		initial_solution_fn=lambda x,y: np.zeros(1),
		bound_method='ttest',
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
	warning_msg = ("Warning: not enough data to run the algorithm.  " 
                   "Returning NSF and failing safety test.") 
	with pytest.warns(UserWarning,match=warning_msg) as excinfo:
		passed_safety,solution = seldonian_algorithm(spec_zeros)

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
				frac_data_in_safety=frac_data_in_safety,
				primary_objective=primary_objective,
				use_builtin_primary_gradient_fn=False,
				parse_trees=parse_trees,
				initial_solution_fn=model_class().fit,
				bound_method='ttest',
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
			passed_safety,solution = seldonian_algorithm(bad_spec)
		error_str = "Optimizer: bad-optimizer is not supported"
		assert error_str in str(excinfo.value)

	bad_optimization_technique = 'bad-opt-technique' 

	bad_spec = SupervisedSpec(
			dataset=dataset,
			model_class=model_class,
			frac_data_in_safety=frac_data_in_safety,
			primary_objective=primary_objective,
			use_builtin_primary_gradient_fn=False,
			parse_trees=parse_trees,
			initial_solution_fn=model_class().fit,
			bound_method='ttest',
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
		passed_safety,solution = seldonian_algorithm(bad_spec)
	error_str = "Optimization technique: bad-opt-technique is not implemented"
	assert error_str in str(excinfo.value)

def test_gpa_data_regression(gpa_regression_dataset):
	""" Test that the gpa regression example runs 
	with a simple non-conflicting constraint. Make
	sure safety test passes and solution is correct.
	"""
	rseed=0
	np.random.seed(rseed) 
	# constraint_strs = ['Mean_Squared_Error - 2.0']
	constraint_strs = ['abs((Mean_Error|[M]) - (Mean_Error|[F])) - 0.1']

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
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		parse_trees=parse_trees,
		initial_solution_fn=model_class().fit,
		bound_method='ttest',
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.01,
			'alpha_lamb'    : 0.01,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 600,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		}
	)

	# Run seldonian algorithm
	passed_safety,solution = seldonian_algorithm(spec)
	assert passed_safety == True
	array_to_compare = np.array(
		[ 4.20776626e-01, -6.68167090e-04,  9.78329737e-04, -1.15722866e-03,
 -1.05315739e-03, -8.12009686e-04,  4.61741069e-03,  3.92829593e-03,
 -1.40006741e-03,  3.41671151e-05])

	assert np.allclose(solution,array_to_compare)

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
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		parse_trees=parse_trees,
		initial_solution_fn=model_class().fit,
		bound_method='ttest',
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : 0.5,
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
	passed_safety,solution = seldonian_algorithm(spec)
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
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		parse_trees=parse_trees,
		initial_solution_fn=model_class().fit,
		bound_method='ttest',
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : 0.5,
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
	passed_safety,solution = seldonian_algorithm(spec)
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
		[-0.07451169, -0.04746089,  0.15604706,
		 0.10953836,  0.08014627,  0.03998193,
		 0.40485252,  0.3045781,  -0.10843437,
		 -0.05772248]),
	'demographic_parity':np.array(
		[-0.07951141, -0.042461, 0.16104631,
		 0.1145382,   0.08514536,  0.04498181,
		 0.39985267,  0.29957834, -0.10343447,
		 -0.05272268]),
	'equalized_odds':np.array(
		[-0.07951143, -0.04246094,  0.16104688,
		 0.1145383,   0.08514607,  0.04498188,
		 0.39985263,  0.2995783,  -0.10343441,
		 -0.05272256]),
	'equal_opportunity':np.array(
		[-0.07951139, -0.04246098,  0.16104682,
		 0.11453824,  0.08514599,  0.04498183,
		 0.39985278,  0.29957895, -0.10343445, 
		 -0.05272261]),
	'predictive_equality':np.array(
		[-0.07951143, -0.04246099,  0.16104644,
		 0.11453823,  0.08514568,  0.04498183,
		 0.39985272,  0.29957838, -0.10343446, 
		 -0.05272268])
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
			frac_data_in_safety=frac_data_in_safety,
			primary_objective=primary_objective,
			use_builtin_primary_gradient_fn=True,
			parse_trees=parse_trees,
			initial_solution_fn=model_class().fit,
			bound_method='ttest',
			optimization_technique='gradient_descent',
			optimizer='adam',
			optimization_hyperparams={
				'lambda_init'   : 0.5,
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
		passed_safety,solution = seldonian_algorithm(spec)
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
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		parse_trees=parse_trees,
		initial_solution_fn=model_class().fit,
		bound_method='ttest',
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : 0.5,
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
	passed_safety,solution = seldonian_algorithm(spec)
	assert passed_safety == True
	print(passed_safety,solution)
	solution_to_compare = np.array(
		[-0.07451169, -0.04746089,  0.15604706,
		 0.10953836,  0.08014627,  0.03998193,
		 0.40485252,  0.3045781,  -0.10843437,
		 -0.05772248])

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
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		parse_trees=parse_trees,
		initial_solution_fn=model_class().fit,
		bound_method='ttest',
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : 0.5,
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
	passed_safety,solution = seldonian_algorithm(spec)
	assert passed_safety == False
	assert solution == 'NSF'

def test_black_box_optimizers(gpa_regression_dataset):
	""" Test that the black box optimizers successfully optimize the GPA 
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

	for optimizer in ['Powell','CG','Nelder-Mead','BFGS','CMA-ES']:
		spec = SupervisedSpec(
			dataset=dataset,
			model_class=model_class,
			frac_data_in_safety=frac_data_in_safety,
			primary_objective=primary_objective,
			use_builtin_primary_gradient_fn=False,
			parse_trees=parse_trees,
			initial_solution_fn=model_class().fit,
			bound_method='ttest',
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
		passed_safety,solution = seldonian_algorithm(spec)

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
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=False,
		custom_primary_gradient_fn=gradient_MSE,
		parse_trees=parse_trees,
		initial_solution_fn=model_class().fit,
		bound_method='ttest',
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : 0.5,
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
	passed_safety,solution = seldonian_algorithm(spec)
	assert passed_safety == True
	array_to_compare = np.array(
		[ 4.17882259e-01, -1.59868384e-04,  6.33766780e-04,  2.64271363e-04,
  3.08303718e-04,  1.01170148e-04,  1.86987938e-03,  1.29098727e-03,
 -3.82405534e-04,  2.29938169e-04])
	assert np.allclose(solution,array_to_compare)

""" RL based tests """

def test_RL_builtin_or_custom_gradient_not_supported():
	""" Test that an error is raised if user tries to 
	use built-in gradient or a custom gradient 
	when doing RL
	"""
	# Load data and metadata
	np.random.seed(0) 
	data_pth = 'static/datasets/RL/gridworld/gridworld3x3_50episodes.csv'
	metadata_pth = 'static/datasets/RL/gridworld/gridworld3x3_metadata.json'

	metadata_dict = load_json(metadata_pth)
	regime = metadata_dict['regime']
	columns = metadata_dict['columns']
				
	include_sensitive_columns = False
	include_intercept_term = False
	frac_data_in_safety = 0.6

	# Model
	model_class = TabularSoftmaxModel

	# RL environment
	RL_environment_name = metadata_dict['RL_environment_name']
	RL_environment_module = importlib.import_module(
		f'seldonian.RL.environments.{RL_environment_name}')
	RL_environment_obj = RL_environment_module.Environment()    

	# Primary objective
	model_instance = model_class(RL_environment_obj)

	primary_objective = model_instance.default_objective
	# Load dataset from file
	loader = DataSetLoader(
		regime=regime)

	dataset = loader.load_RL_dataset_from_csv(
		filename=data_pth,
		metadata_filename=metadata_pth)
	
	constraint_strs = ['-0.25 - J_pi_new'] 
	
	deltas = [0.05]

	# For each constraint, make a parse tree
	parse_trees = []
	for ii in range(len(constraint_strs)):
		constraint_str = constraint_strs[ii]

		delta = deltas[ii]
		# Create parse tree object
		parse_tree = ParseTree(delta=delta,regime='RL',
		sub_regime='all')

		# Fill out tree
		parse_tree.create_from_ast(constraint_str)
		# assign deltas for each base node
		# use equal weighting for each base node
		parse_tree.assign_deltas(weight_method='equal')

		# Assign bounds needed on the base nodes
		parse_tree.assign_bounds_needed()
		
		parse_trees.append(parse_tree)

	# # Create spec object
	spec = RLSpec(
		dataset=dataset,
		model_class=model_class,
		frac_data_in_safety=0.8,
		use_builtin_primary_gradient_fn=True,
		primary_objective=primary_objective,
		parse_trees=parse_trees,
		RL_environment_obj=RL_environment_obj,
		initial_solution_fn=None,
		bound_method='ttest',
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : 0.5,
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 20,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		},
		regularization_hyperparams={'reg_coef':0.1},
		normalize_returns=False,
	)

	# Run seldonian algorithm, making sure we capture error
	error_str = ("Using a builtin primary objective gradient"
				" is not yet supported for regimes other"
				" than supervised learning")
	with pytest.raises(NotImplementedError) as excinfo:
		passed_safety,solution = seldonian_algorithm(spec)
		
	assert error_str in str(excinfo.value)

	# # Create spec object
	spec2 = RLSpec(
		dataset=dataset,
		model_class=model_class,
		frac_data_in_safety=0.8,
		use_builtin_primary_gradient_fn=False,
		custom_primary_gradient_fn=lambda x: x,
		primary_objective=primary_objective,
		parse_trees=parse_trees,
		RL_environment_obj=RL_environment_obj,
		initial_solution_fn=None,
		bound_method='ttest',
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : 0.5,
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 20,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		},
		regularization_hyperparams={'reg_coef':0.1},
		normalize_returns=False,
	)

	# Run seldonian algorithm, making sure we capture error
	error_str2 = ("Using a provided primary objective gradient"
				" is not yet supported for regimes other"
				" than supervised learning")
	with pytest.raises(NotImplementedError) as excinfo2:
		passed_safety,solution = seldonian_algorithm(spec2)
		
	assert error_str2 in str(excinfo2.value)
	
def test_RL_gridworld_gradient_descent(RL_gridworld_dataset):
	""" Test that the RL gridworld example runs 
	with a simple performance improvement constraint. Make
	sure safety test passes and solution is correct.
	"""
	# Load data and metadata
	rseed=99
	np.random.seed(rseed)
	constraint_strs = ['-0.25 - J_pi_new']
	deltas = [0.05]
	
	(dataset,model_class,
		primary_objective,
		parse_trees,
		RL_environment_obj) = RL_gridworld_dataset(
			constraint_strs=constraint_strs,
			deltas=deltas)

	frac_data_in_safety = 0.6
	# Create spec object
	spec = RLSpec(
		dataset=dataset,
		model_class=model_class,
		frac_data_in_safety=frac_data_in_safety,
		use_builtin_primary_gradient_fn=False,
		primary_objective=primary_objective,
		parse_trees=parse_trees,
		RL_environment_obj=RL_environment_obj,
		initial_solution_fn=None,
		bound_method='ttest',
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 15,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		},
		normalize_returns=False,
	)

	# # Run seldonian algorithm
	passed_safety,solution = seldonian_algorithm(spec)
	assert passed_safety == True
	array_to_compare = np.array(
	   [ 0.14021112, -0.14216886, -0.13786284, -0.14577261,  0.14013514,  0.13784917,
 -0.14094687, -0.13881718,  0.13240779,  0.1411048,  -0.13954501,  0.1277282,
  0.14035668, -0.14190627,  0.05335083, -0.13960011,  0.14010328, -0.14035336,
  0.13488608,  0.14067562, -0.14114499,  0.14104931, -0.14068246,  0.14011436,
 -0.13970751,  0.14057111, -0.1423285,   0.14027992, -0.13829985, -0.1416236,
 -0.13949036,  0.14042179])
	assert np.allclose(solution,array_to_compare)

def test_RL_gridworld_black_box(RL_gridworld_dataset):
	""" Test that the RL gridworld example runs 
	with a simple performance improvement constraint. Make
	sure safety test passes and solution is correct.
	"""
	# Load data and metadata
	rseed=99
	np.random.seed(rseed)
	constraint_strs = ['-0.25 - J_pi_new']
	deltas = [0.05]
	
	(dataset,model_class,
		primary_objective,
		parse_trees,
		RL_environment_obj) = RL_gridworld_dataset(
			constraint_strs=constraint_strs,
			deltas=deltas)

	frac_data_in_safety = 0.6
	# Create spec object
	spec = RLSpec(
		dataset=dataset,
		model_class=model_class,
		frac_data_in_safety=frac_data_in_safety,
		use_builtin_primary_gradient_fn=False,
		primary_objective=primary_objective,
		parse_trees=parse_trees,
		RL_environment_obj=RL_environment_obj,
		initial_solution_fn=None,
		bound_method='ttest',
		optimization_technique='barrier_function',
		optimizer='CMA-ES',
		optimization_hyperparams={
				'maxiter'   : 10,
				'seed':rseed,
				'hyper_search'  : None,
				'verbose'       : True,
			},
		normalize_returns=False,
	)

	# # Run seldonian algorithm
	passed_safety,solution = seldonian_algorithm(spec)
	assert passed_safety == True
	array_to_compare = np.array(
	   [-0.01478694, -1.52299897,  0.60606259, -0.0475612,  -0.13658868,  0.33656384,
 -0.53947718, -0.2725281,   0.44593005,  0.86750493,  0.79728041,  0.10156789,
  0.2646691,  -0.66778523,  0.6754071,   0.29783526,  0.49140432, -0.54893211,
 -0.6835978,   0.93589933, -0.97537624,  1.19449046, -0.724231,    0.40491889,
 -0.6061301,  -0.22444457, -0.76379049,  0.51380723,  0.86515137,  0.2919332,
 -0.31993747,  0.27008568])
	assert np.allclose(solution,array_to_compare)
