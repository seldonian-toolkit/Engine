""" Module for building the specification object needed to run Seldonian algorithms """
import os
import importlib 

from seldonian.utils.io_utils import load_supervised_metadata,save_pickle
from seldonian.models.models import *
from seldonian.models import objectives
from seldonian.parse_tree.parse_tree import (
	make_parse_trees_from_constraints)


class Spec(object):
	"""Base class for specification object required to
	run the Seldonian algorithm

	:param dataset: The dataset object containing safety data
	:type dataset: :py:class:`.DataSet` object
	:param model: The :py:class:`.SeldonianModel` object
	:param frac_data_in_safety: Fraction of data used in safety test.
		The remaining fraction will be used in candidate selection
	:type frac_data_in_safety: float
	:param primary_objective: The objective function that would
		be solely optimized in the absence of behavioral constraints,
		i.e. the loss function
	:type primary_objective: function or class method
	:param initial_solution_fn: Function to provide 
		initial model weights in candidate selection 
	:type initial_solution_fn: function
	:param parse_trees: List of parse tree objects containing the 
			behavioral constraints
	:type parse_trees: List(:py:class:`.ParseTree` objects)
	:param use_builtin_primary_gradient_fn: Whether to use the built-in
		function for the gradient of the primary objective, 
		if one exists. If False, uses autograd
	:type use_builtin_primary_gradient_fn: bool, defaults to True
	:param custom_primary_gradient_fn: A function for computing 
		the gradient of the primary objective. If None,
		falls back on builtin function or autograd
	:type custom_primary_gradient_fn: function, defaults to None 
	:param bound_method: 
		The statistical method for calculating the confidence bounds
	:type bound_method: str, defaults to 'ttest'
	:param optimization_technique: The method for optimization during 
		candidate selection. E.g. 'gradient_descent', 'barrier_function'
	:type optimization_technique: str, defaults to 'gradient_descent'
	:param optimizer: The string name of the optimizer used 
		during candidate selection
	:type optimizer: str, defaults to 'adam'
	:param optimization_hyperparams: Hyperparameters for 
		optimization during candidate selection. See :ref:`candidate_selection`.
	:type optimization_hyperparams: dict
	:param regularization_hyperparams: Hyperparameters for 
		regularization during candidate selection. See :ref:`candidate_selection`.
	:type regularization_hyperparams: dict
	:ivar usable_opt_dict: Shows which optimizers are acceptable
		for a given optimization technique
	:vartype usable_opt_dict: dict
	"""

	def __init__(
		self,
		dataset,
		model,
		frac_data_in_safety,
		primary_objective,
		initial_solution_fn,
		parse_trees,
		base_node_bound_method_dict={},
		use_builtin_primary_gradient_fn=True,
		custom_primary_gradient_fn=None,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : 0.5,
			'alpha_theta'   : 0.005,
			'alpha_lamb'    : 0.005,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'num_iters'     : 200,
			'gradient_library': 'autograd',
			'hyper_search'  : None,
			'verbose'       : True,
		},
		regularization_hyperparams={}
		):
		self.dataset = dataset
		self.model = model 
		self.frac_data_in_safety = frac_data_in_safety
		self.primary_objective = primary_objective
		self.initial_solution_fn = initial_solution_fn
		self.use_builtin_primary_gradient_fn=use_builtin_primary_gradient_fn
		self.custom_primary_gradient_fn = custom_primary_gradient_fn
		self.parse_trees = parse_trees
		self.base_node_bound_method_dict = base_node_bound_method_dict
		self.optimization_technique = optimization_technique
		self.optimizer = optimizer
		self.optimization_hyperparams = optimization_hyperparams
		self.regularization_hyperparams = regularization_hyperparams


class SupervisedSpec(Spec):
	""" Specification object for running Supervised learning
	Seldonian algorithms 

	:param dataset: The dataset object containing safety data
	:type dataset: :py:class:`.DataSet` object

	:param model: The SeldonianModel object

	:param parse_trees: List of parse tree objects containing the 
			behavioral constraints
	:param sub_regime: "classification" or "regression"
	
	:param frac_data_in_safety: Fraction of data used in safety test.
		The remaining fraction will be used in candidate selection
	:type frac_data_in_safety: float

	:param primary_objective: The objective function that would
		be solely optimized in the absence of behavioral constraints,
		i.e. the loss function

	:param initial_solution_fn: Function to provide 
		initial model weights in candidate selection
	
	:param use_builtin_primary_gradient_fn: Whether to use the built-in
		function for the gradient of the primary objective, 
		if one exists. If False, uses autograd
	:type use_builtin_primary_gradient_fn: bool, defaults to True

	:param custom_primary_gradient_fn: A function for computing 
		the gradient of the primary objective. If None,
		falls back on builtin function or autograd
	:type custom_primary_gradient_fn: function, defaults to None 

	:param bound_method: 
		The statistical method for calculating the confidence bounds
	:type bound_method: str, defaults to 'ttest'

	:param optimization_technique: The method for optimization during 
		candidate selection. E.g. 'gradient_descent', 'barrier_function'
	:type optimization_technique: str, defaults to 'gradient_descent'

	:param optimizer: The string name of the optimizer used 
		during candidate selection
	:type optimizer: str, defaults to 'adam'

	:param optimization_hyperparams: Hyperparameters for 
		optimization during candidate selection. See :ref:`candidate_selection`.
	:type optimization_hyperparams: dict

	:param regularization_hyperparams: Hyperparameters for 
		regularization during candidate selection. See :ref:`candidate_selection`.
	:type regularization_hyperparams: dict

	:ivar usable_opt_dict: Shows which optimizers are acceptable
		for a given optimization technique
	:vartype usable_opt_dict: dict

	"""
	def __init__(self,
		dataset,
		model,
		parse_trees,
		sub_regime,
		primary_objective=None,
		initial_solution_fn=None,
		frac_data_in_safety=0.6,
		base_node_bound_method_dict={},
		use_builtin_primary_gradient_fn=True,
		custom_primary_gradient_fn=None,
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
		regularization_hyperparams={},
		):
		super().__init__(
			dataset=dataset,
			model=model,
			parse_trees=parse_trees,
			primary_objective=primary_objective,
			initial_solution_fn=initial_solution_fn,
			frac_data_in_safety=frac_data_in_safety,
			base_node_bound_method_dict=base_node_bound_method_dict,
			use_builtin_primary_gradient_fn=use_builtin_primary_gradient_fn,
			custom_primary_gradient_fn=custom_primary_gradient_fn,
			optimization_technique=optimization_technique,
			optimizer=optimizer,
			optimization_hyperparams=optimization_hyperparams,
			regularization_hyperparams=regularization_hyperparams)
		self.sub_regime = sub_regime


class RLSpec(Spec):
	""" Specification object for running RL Seldonian algorithms

	:param dataset: The dataset object containing safety data
	:type dataset: :py:class:`.DataSet` object

	:param model: The :py:class:`.SeldonianModel` object

	:param frac_data_in_safety: Fraction of data used in safety test.
		The remaining fraction will be used in candidate selection
	:type frac_data_in_safety: float

	:param primary_objective: The objective function that would
		be solely optimized in the absence of behavioral constraints,
		i.e. the loss function
	:type primary_objective: function or class method

	:param initial_solution_fn: Function to provide 
		initial model weights in candidate selection 
	:type initial_solution_fn: function

	:param parse_trees: List of parse tree objects containing the 
			behavioral constraints
	:type parse_trees: List(:py:class:`.ParseTree` objects)
	
	:param RL_environment_obj: Environment class from an RL
		module in this library (see :py:mod:`seldonian.RL.environments`)

	:param use_builtin_primary_gradient_fn: Whether to use the built-in
		function for the gradient of the primary objective, 
		if one exists. If False, uses autograd
	:type use_builtin_primary_gradient_fn: bool, defaults to True

	:param custom_primary_gradient_fn: A function for computing 
		the gradient of the primary objective. If None,
		falls back on builtin function or autograd
	:type custom_primary_gradient_fn: function, defaults to None 

	:param bound_method: 
		The statistical method for calculating the confidence bounds
	:type bound_method: str, defaults to 'ttest'

	:param optimization_technique: The method for optimization during 
		candidate selection. E.g. 'gradient_descent', 'barrier_function'
	:type optimization_technique: str, defaults to 'gradient_descent'

	:param optimizer: The string name of the optimizer used 
		during candidate selection
	:type optimizer: str, defaults to 'adam'

	:param optimization_hyperparams: Hyperparameters for 
		optimization during candidate selection. See 
		:ref:`candidate_selection`.
	:type optimization_hyperparams: dict

	:param regularization_hyperparams: Hyperparameters for 
		regularization during candidate selection. See 
		:ref:`candidate_selection`.
	:type regularization_hyperparams: dict

	:param normalize_returns: Whether to normalize the returns to [0,1]
	:type normalize_returns: bool, defaults to False

	:ivar usable_opt_dict: Shows which optimizers are acceptable
		for a given optimization technique
	:vartype usable_opt_dict: dict

	"""
	def __init__(
		self,
		dataset,
		model,
		frac_data_in_safety,
		primary_objective,
		initial_solution_fn,
		parse_trees,
		base_node_bound_method_dict={},
		use_builtin_primary_gradient_fn=True,
		custom_primary_gradient_fn=None,
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
		regularization_hyperparams={},
		):

		super().__init__(
			dataset=dataset,
			model=model,
			frac_data_in_safety=frac_data_in_safety,
			primary_objective=primary_objective,
			initial_solution_fn=initial_solution_fn,
			parse_trees=parse_trees,
			base_node_bound_method_dict=base_node_bound_method_dict,
			use_builtin_primary_gradient_fn=use_builtin_primary_gradient_fn,
			custom_primary_gradient_fn=custom_primary_gradient_fn,
			optimization_technique=optimization_technique,
			optimizer=optimizer,
			optimization_hyperparams=optimization_hyperparams,
			regularization_hyperparams=regularization_hyperparams)

def createSupervisedSpec(
	dataset,
	metadata_pth,
	constraint_strs,
	deltas,
	save_dir,
	save=True,
	verbose=False):
	"""Convenience function for creating SupervisedSpec object. 
	Uses default model.
	Saves spec.pkl file in save_dir

	:param dataset: The dataset object containing data and metadata
	:type dataset: :py:class:`.DataSet`
	:param metadata_pth: Path to metadata file
	:type metadata_pth: str
	:param constraint_strs: Constraint strings 
	:type constraint_strs: List(str)
	:param deltas: Confidence thresholds
	:type deltas: List(float)
	:param save_dir: Directory where to save the spec.pkl file
	:type save_dir: str
	:param verbose: Flag to control verbosity 
	:type verbose: bool
	"""
	# Load metadata
	(regime, sub_regime, columns,
        sensitive_columns) = load_supervised_metadata(metadata_pth)

	assert regime == 'supervised_learning'

	if sub_regime == 'regression':
		model = LinearRegressionModel()
		primary_objective = objectives.Mean_Squared_Error
	elif sub_regime == 'classification':
		model = LogisticRegressionModel()
		primary_objective = objectives.logistic_loss

	parse_trees = make_parse_trees_from_constraints(
		constraint_strs,
		deltas,
		regime='supervised_learning',
		sub_regime=sub_regime,
		columns=columns,
		delta_weight_method='equal')

	# Save spec object, using defaults where necessary
	spec = SupervisedSpec(
		dataset=dataset,
		model=model,
		frac_data_in_safety=0.6,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		parse_trees=parse_trees,
		sub_regime=sub_regime,
		initial_solution_fn=model.fit,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : 0.5,
            'alpha_theta'   : 0.01,
            'alpha_lamb'    : 0.01,
            'beta_velocity' : 0.9,
            'beta_rmsprop'  : 0.95,
            'num_iters'     : 1000,
            'gradient_library': "autograd",
            'hyper_search'  : None,
            'verbose'       : True,
		},
	)

	spec_save_name = os.path.join(save_dir, 'spec.pkl')
	if save:
		save_pickle(spec_save_name,spec,verbose=verbose)


def createRLSpec(
	dataset,
	policy,
	constraint_strs,
	deltas,
	env_kwargs={},
	frac_data_in_safety=0.6,
	initial_solution_fn=None,
	use_builtin_primary_gradient_fn=False,
	save=False,
	save_dir='.',
	verbose=False):
	"""Convenience function for creating RLSpec object. 
	Saves spec.pkl file in save_dir

	:type dataset: :py:class:`.DataSet`
	:type policy: :py:class:`.Policy`
	:param constraint_strs: List of constraint strings 
	:param deltas: List of confidence thresholds
	:param save_dir: Directory in which to save the spec.pkl file
	:param env_kwargs: Kwargs passed to RL_model pertaining to environment, 
		such as gamma, the discount factor 
	:type env_kwargs: dict
	:param verbose: Flag to control verbosity 
	:type verbose: bool
	"""
	from seldonian.RL.RL_model import RL_model
	# Define primary objective
	primary_objective = objectives.IS_estimate

	# Create parse trees
	parse_trees = make_parse_trees_from_constraints(
		constraint_strs,
		deltas,
		regime='reinforcement_learning',
		sub_regime='all',
		delta_weight_method='equal')

	model = RL_model(policy=policy,env_kwargs=env_kwargs)
	# Save spec object, using defaults where necessary
	spec = RLSpec(
		dataset=dataset,
		model=model,
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=use_builtin_primary_gradient_fn,
		parse_trees=parse_trees,
		initial_solution_fn=initial_solution_fn,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init': 0.5,
			'alpha_theta': 0.005,
			'alpha_lamb': 0.005,
			'beta_velocity': 0.9,
			'beta_rmsprop': 0.95,
			'num_iters': 30,
			'hyper_search': None,
			'gradient_library': 'autograd',
			'verbose': verbose,
		},
		regularization_hyperparams={},
	)

	if save:
		spec_save_name = os.path.join(save_dir, 'spec.pkl')
		save_pickle(spec_save_name,spec,verbose=verbose)
	return spec

