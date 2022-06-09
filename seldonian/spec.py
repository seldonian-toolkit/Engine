""" Module for building the specification object needed to run Seldonian algorithms """

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
		use_builtin_primary_gradient_fn=True,
		custom_primary_gradient_fn=None,
		bound_method='ttest',
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={},
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
		self.bound_method = bound_method
		self.optimization_technique = optimization_technique
		self.optimizer = optimizer
		self.optimization_hyperparams = optimization_hyperparams
		self.regularization_hyperparams = regularization_hyperparams

		self.usable_opt_dict = {
			'gradient_descent' : ['adam'],
			'barrier_function': ['Powell','CG','Nelder-Mead','BFGS','CMA-ES']
		}

		acceptable_optimizers = self.usable_opt_dict[self.optimization_technique]
		if self.optimizer not in acceptable_optimizers:
			raise NotImplementedError(
				f"Optimizer: {self.optimizer} is not an acceptable "
				f"optimizer for optimization_technique: "
				f"{self.optimization_technique}. Must be one of: "
				f"{acceptable_optimizers}")

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
		frac_data_in_safety,
		primary_objective,
		initial_solution_fn,
		parse_trees,
		use_builtin_primary_gradient_fn=True,
		custom_primary_gradient_fn=None,
		bound_method='ttest',
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={},
		regularization_hyperparams={},
		):
		super().__init__(dataset,model_class,
			frac_data_in_safety,
			primary_objective,
			initial_solution_fn,
			parse_trees,
			use_builtin_primary_gradient_fn,
			custom_primary_gradient_fn,
			bound_method,
			optimization_technique,
			optimizer,
			optimization_hyperparams,
			regularization_hyperparams)

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
		use_builtin_primary_gradient_fn=True,
		custom_primary_gradient_fn=None,
		bound_method='ttest',
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={},
		regularization_hyperparams={},
		normalize_returns=False
		):
		super().__init__(dataset,model_class,
			frac_data_in_safety,
			primary_objective,
			initial_solution_fn,
			parse_trees,
			use_builtin_primary_gradient_fn,
			custom_primary_gradient_fn,
			bound_method,
			optimization_technique,
			optimizer,
			optimization_hyperparams,
			regularization_hyperparams)
		self.RL_environment_obj = RL_environment_obj
		self.normalize_returns = normalize_returns

