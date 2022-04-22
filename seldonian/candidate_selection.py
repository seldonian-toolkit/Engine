import numpy as np

class CandidateSelection(object):
	def __init__(self,
		model,
		candidate_dataset,
		n_safety,
		parse_trees,
		primary_objective,
		optimization_technique='barrier_function',
		optimizer='Powell',
		initial_solution=None,
		regime='supervised',
		**kwargs):
		self.regime = regime
		self.model = model
		self.candidate_dataset = candidate_dataset
		self.n_safety = n_safety
		if self.regime == 'supervised':
			# Separate features from label
			label_column = candidate_dataset.label_column
			self.labels = self.candidate_dataset.df[label_column]
			self.features = self.candidate_dataset.df.loc[:,
				self.candidate_dataset.df.columns != label_column]
			if not candidate_dataset.include_sensitive_columns:
				self.features = self.features.drop(
					columns=self.candidate_dataset.sensitive_column_names)
			if candidate_dataset.include_intercept_term:
				self.features.insert(0,'offset',1.0) # inserts a column of 1's
		self.parse_trees = parse_trees
		self.primary_objective = primary_objective # must accept theta, features, labels
		self.optimization_technique = optimization_technique
		self.optimizer = optimizer
		self.initial_solution = initial_solution
		self.candidate_solution = None

	def run(self,**kwargs):

		# print("initial solution is:",initial_solution)
		if self.optimizer in ['Powell','CG','Nelder-Mead','BFGS',]:
			from scipy.optimize import minimize 
			res = minimize(
				self.candidate_objective,
				x0=self.initial_solution,
				method=self.optimizer,
				options=kwargs['minimizer_options'], 
				args=())
			
			candidate_solution=res.x

		elif self.optimizer == 'CMA-ES':
			# from seldonian.cmaes import minimize
			import cma
			n_iters=500
			options = {'tolfun':1e-5, 'maxiter':n_iters}

			es = cma.CMAEvolutionStrategy(self.initial_solution, 0.5,options)
			# es = cma.CMAEvolutionStrategy(np.zeros_like(initial_solution), 0.5,options)
			
			es.optimize(self.candidate_objective)
			# print(es.result_pretty())
			candidate_solution=es.result.xbest
			# 
			# N=self.features.shape[1]
			# N = len(self.initial_solution)
			# candidate_solution = minimize(N=N,
			# 	# lamb=int(4+np.floor(3*np.log(N))),
			# 	lamb=20,
			# 	initial_solution=np.zeros_like(self.initial_solution),
			# 	objective=self.candidate_objective)
		else:
			raise NotImplementedError(
				f"Optimizer {self.optimizer} is not supported")
		# Reset parse tree base node dicts, 
		# including data and datasize attributes
		for pt in self.parse_trees:
			pt.reset_base_node_dict(reset_data=True)
		# print(candidate_solution)

		# Unset data and datasize on base nodes
		# Return the candidate solution we believe will pass the safety test
		return candidate_solution

	def candidate_objective(self,theta):
		# Get the primary objective evaluated at the given theta
		# and the entire candidate dataset
		# print("theta is:", theta)
		if self.regime == 'supervised':
			result = self.primary_objective(self.model, theta, 
				self.features, self.labels)
		elif self.regime == 'RL':
			# Want to maximize the importance weight so minimize negative importance weight
			result = -1.0*self.primary_objective(self.model,theta,
				self.candidate_dataset) 
		# print("Primary objective eval is:", result)
		# Prediction of what the safety test will return. 
		# Initialized to pass
		predictSafetyTest = True     
		for tree_i,pt in enumerate(self.parse_trees): 
			# before we propagate, reset the bounds on all base nodes
			pt.reset_base_node_dict()

			pt.propagate_bounds(
				theta=theta,
				dataset=self.candidate_dataset,
				model=self.model,
				bound_method='ttest',
				branch='candidate_selection',
				n_safety=self.n_safety,
				regime=self.regime)

			# Check if the i-th behavioral constraint is satisfied
			upper_bound = pt.root.upper 
			# print(f"Upper_bound on ghat: {upper_bound}") 
			if self.optimization_technique == 'barrier_function':
				if upper_bound > 0.0: # If the current constraint was not satisfied, the safety test failed
					# If up until now all previous constraints passed,
					# then we need to predict that the test will fail
					# and potentially add a penalty to the objective
					if predictSafetyTest:
						# Set this flag to indicate that we don't think the safety test will pass
						predictSafetyTest = False  

						# Put a barrier in the objective. Any solution 
						# that we think will fail the safety test 
						# will have a large cost associated with it

						result = 100000.0    

					# Add a shaping to the objective function that will 
					# push the search toward solutions that will pass 
					# the prediction of the safety test
					result = result + upper_bound
		print(f"Result = {result}")
		
		# title = f'Parse tree'
		# graph = pt.make_viz(title)
		# graph.attr(fontsize='12')
		# graph.view() # to open it as a pdf
		# input("End of optimzer iteration")
		return result
