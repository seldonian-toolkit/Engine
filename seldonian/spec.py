""" Module for building the specification object needed to run Seldonian algorithms """
import os
import importlib 

from seldonian.utils.io_utils import load_json,save_pickle
from seldonian.models.models import *
from seldonian.parse_tree.parse_tree import (
	make_parse_trees_from_constraints)


class Spec(object):
	"""Base class for specification object required to
	run the Seldonian algorithm

	:param dataset: The dataset object containing safety data
	:type dataset: :py:class:`.DataSet` object
	:param model_class: The model class for the Seldonian model,
		not an instance of the model.
	:type model_class: :py:class:`.SeldonianModel` or child class
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
		model_class,
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
		self.model_class = model_class 
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

	:param model_class: The model class for the Seldonian model,
		not an instance of the model.
	:type model_class: :py:class:`.SeldonianModel` or child class

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
	def __init__(self,
		dataset,
		model_class,
		parse_trees,
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
			model_class=model_class,
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


class RLSpec(Spec):
	""" Specification object for running RL Seldonian algorithms

	:param dataset: The dataset object containing safety data
	:type dataset: :py:class:`.DataSet` object

	:param model_class: The model class for the Seldonian model,
		not an instance of the model.
	:type model_class: :py:class:`.SeldonianModel` or child class

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
		model_class,
		frac_data_in_safety,
		primary_objective,
		initial_solution_fn,
		parse_trees,
		RL_environment_obj,
		RL_agent_obj,
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
		normalize_returns=False
		):
		super().__init__(
			dataset,
			model_class,
			frac_data_in_safety,
			primary_objective,
			initial_solution_fn,
			parse_trees,
			base_node_bound_method_dict,
			use_builtin_primary_gradient_fn,
			custom_primary_gradient_fn,
			optimization_technique,
			optimizer,
			optimization_hyperparams,
			regularization_hyperparams)
		self.RL_environment_obj = RL_environment_obj
		self.RL_agent_obj = RL_agent_obj
		self.normalize_returns = normalize_returns


def createRLSpec(
	dataset,
	metadata_pth,
	agent,
	constraint_strs,
	deltas,
	save_dir,
	env_kwargs={},
	verbose=False):
	"""Convenience function for creating RLSpec object. 
	Saves spec.pkl file in save_dir

	:param dataset: The dataset object containing data and metadata
	:type dataset: :py:class:`.DataSet`
	:param metadata_pth: Path to metadata file
	:type metadata_pth: str
	:param agent: The agent object 
	:type agent: :py:class:`.Agent`
	:param constraint_strs: Constraint strings 
	:type constraint_strs: List(str)
	:param deltas: Confidence thresholds
	:type deltas: List(float)
	:param save_dir: Directory where to save the spec.pkl file
	:type save_dir: str
	:param env_kwargs: Kwargs passed to Environment object upon creation
	:type env_kwargs: dict
	:param verbose: Flag to control verbosity 
	:type verbose: bool
	"""
	# Load metadata
	metadata_dict = load_json(metadata_pth)
	# Create RL environment environment 
	RL_module_name = metadata_dict['RL_module_name']
	RL_environment_module = importlib.import_module(
		f'seldonian.RL.environments.{RL_module_name}')
	RL_env_class_name = metadata_dict['RL_class_name']

	RL_environment_obj = getattr(
		RL_environment_module, RL_env_class_name)(**env_kwargs)

	RL_model_instance = RL_model(agent,RL_environment_obj)
	primary_objective = RL_model_instance.sample_IS_estimate

	parse_trees = make_parse_trees_from_constraints(
		constraint_strs,
		deltas,
		regime='reinforcement_learning',
		sub_regime='all',
		delta_weight_method='equal')

	# Save spec object, using defaults where necessary
	spec = RLSpec(
		dataset=dataset,
		model_class=RL_model,
		frac_data_in_safety=0.6,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=False,
		parse_trees=parse_trees,
		RL_environment_obj=RL_environment_obj,
		RL_agent_obj=agent,
		initial_solution_fn=None,
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
			'verbose': True,
		},
		regularization_hyperparams={},
		normalize_returns=False,
	)

	spec_save_name = os.path.join(save_dir, 'spec.pkl')
	save_pickle(spec_save_name,spec,verbose=verbose)

def createSupervisedSpec(
	dataset,
	metadata_pth,
	constraint_strs,
	deltas,
	save_dir,
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
	metadata_dict = load_json(metadata_pth)
	# Create RL environment environment 
	regime = metadata_dict['regime']
	assert regime == 'supervised_learning'
	sub_regime = metadata_dict['sub_regime']

	if sub_regime == 'regression':
		model_class = LinearRegressionModel
	elif sub_regime == 'classification':
		model_class = LogisticRegressionModel

	primary_objective = model_class().default_objective

	parse_trees = make_parse_trees_from_constraints(
		constraint_strs,
		deltas,
		regime='supervised_learning',
		sub_regime=sub_regime,
		delta_weight_method='equal')

	# Save spec object, using defaults where necessary
	spec = SupervisedSpec(
		dataset=dataset,
		model_class=model_class,
		frac_data_in_safety=0.6,
		primary_objective=primary_objective,
		use_builtin_primary_gradient_fn=True,
		parse_trees=parse_trees,
		initial_solution_fn=model_class().fit,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init': 0.5,
			'alpha_theta': 0.005,
			'alpha_lamb': 0.005,
			'beta_velocity': 0.9,
			'beta_rmsprop': 0.95,
			'num_iters': 500,
			'hyper_search': None,
			'gradient_library': 'autograd',
			'verbose': True,
		},
		regularization_hyperparams={},
		normalize_returns=False,
	)

	spec_save_name = os.path.join(save_dir, 'spec.pkl')
	save_pickle(spec_save_name,spec,verbose=verbose)


