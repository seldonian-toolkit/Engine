class Spec(object):
	""" 

	Specification object for running Seldonian algorithms 

	:param model_class: The model class for the Seldonian model,
		not an instance of the model
	
	:param use_builtin_primary_gradient_fn: Whether to use the built-in
		function for the gradient of the primary objective, 
		if one exists. If False, uses autograd
    :type use_builtin_primary_gradient_fn: bool

    :param custom_primary_gradient_fn: A function for computing 
    	the gradient of the primary objective. If None,
    	falls back on builtin function or autograd
    :type custom_primary_gradient_fn: function taking data 
    	and model weights and outputting the derivative of
    	the primary objective w.r.t model weights as a scalar 

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
		assign_delta_weight_method='equal',
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
		self.parse_trees = parse_trees
		self.assign_delta_weight_method = assign_delta_weight_method
		self.bound_method = bound_method
		self.optimization_technique = optimization_technique
		self.optimizer = optimizer
		self.optimization_hyperparams = optimization_hyperparams
		self.regularization_hyperparams = regularization_hyperparams

		self.usable_opt_dict = {
			'gradient_descent' : ['adam'],
			'black_box_barrier': ['Powell','CG','Nelder-Mead','BFGS']
		}

		acceptable_optimizers = self.usable_opt_dict[self.optimization_technique]
		if self.optimizer not in acceptable_optimizers:
			raise RuntimeError(
				f"Optimizer: {self.optimizer} is not an acceptable "
				f"optimizer for optimization_technique: "
				f"{self.optimization_technique}. Must be one of: "
				f"{acceptable_optimizers}")

class SupervisedSpec(Spec):
	""" 

	Specification object for running Supervised learning
	Seldonian Algorithms 

	:param model_class: The model class for the Seldonian model,
		not an instance of the model
	
	:param use_builtin_primary_gradient_fn: Whether to use the built-in
		function for the gradient of the primary objective, 
		if one exists. If False, uses autograd
    :type use_builtin_primary_gradient_fn: bool

    :param custom_primary_gradient_fn: A function for computing 
    	the gradient of the primary objective. If None,
    	falls back on builtin function or autograd
    :type custom_primary_gradient_fn: function taking data 
    	and model weights and outputting the derivative of
    	the primary objective w.r.t model weights as a scalar 

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
		assign_delta_weight_method='equal',
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
			assign_delta_weight_method,
			bound_method,
			optimization_technique,
			optimizer,
			optimization_hyperparams,
			regularization_hyperparams)


class RLSpec(Spec):
	""" 

	Specification object for running RL Seldonian Algorithm 

	:param model_class: The model class for the Seldonian model,
		not an instance of the model
	
	:param use_builtin_primary_gradient_fn: Whether to use the built-in
		function for the gradient of the primary objective, 
		if one exists. If False, uses autograd
    :type use_builtin_primary_gradient_fn: bool

    :param custom_primary_gradient_fn: A function for computing 
    	the gradient of the primary objective. If None,
    	falls back on builtin function or autograd
    :type custom_primary_gradient_fn: function taking data 
    	and model weights and outputting the derivative of
    	the primary objective w.r.t model weights as a scalar 

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
		assign_delta_weight_method='equal',
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
			assign_delta_weight_method,
			bound_method,
			optimization_technique,
			optimizer,
			optimization_hyperparams,
			regularization_hyperparams)
		self.RL_environment_obj = RL_environment_obj
		self.normalize_returns = normalize_returns

