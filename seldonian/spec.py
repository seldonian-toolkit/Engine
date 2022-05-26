class Spec(object):
	""" 

	Specification object for running Seldonian Algorithm 

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
		model,
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
		self.model = model 
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

