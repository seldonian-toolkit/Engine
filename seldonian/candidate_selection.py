import os, pickle
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd

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
		write_logfile=False,
		**kwargs):
		self.regime = regime
		self.model = model
		self.candidate_dataset = candidate_dataset
		self.n_safety = n_safety
		if self.regime == 'supervised':
			# To get the initial solution we may need to do a model fit
			# For that reason we need the data in fittable form

			# Separate features from label
			label_column = candidate_dataset.label_column
			self.labels = self.candidate_dataset.df[label_column]
			self.features = self.candidate_dataset.df.loc[:,
				self.candidate_dataset.df.columns != label_column]

			if not candidate_dataset.include_sensitive_columns:
				self.features = self.features.drop(
					columns=self.candidate_dataset.sensitive_column_names)
			
			if self.candidate_dataset.scale_features:
				print("Scaling candidate dataset features")
				scaler = kwargs['scaler']
				self.scaler = scaler
				self.features = pd.DataFrame(self.scaler.transform(self.features))
		
			if candidate_dataset.include_intercept_term:
				self.features.insert(0,'offset',1.0) # inserts a column of 1's
		
		self.parse_trees = parse_trees
		self.primary_objective = primary_objective # must accept theta, features, labels
		self.optimization_technique = optimization_technique
		self.optimizer = optimizer
		self.initial_solution = initial_solution
		self.candidate_solution = None
		self.write_logfile = write_logfile
		
		if 'reg_coef' in kwargs:
			self.reg_coef = kwargs['reg_coef']

	def run(self,**kwargs):
		# print("initial solution is:",initial_solution)
		if self.optimization_technique == 'gradient_descent':
			minimizer_options = kwargs['minimizer_options']
			if 'num_iters' in minimizer_options:
				num_iters = minimizer_options['num_iters']
			else:
				num_iters=300

			from seldonian.gradient_descent import gradient_descent_adam

			res = gradient_descent_adam(
				primary_objective=self.evaluate_primary_objective,
				upper_bound_function=self.get_constraint_upper_bound,
				alpha_theta=0.005,
			    alpha_lamb=0.005,
			    beta_velocity=0.9,
			    beta_rmsprop=0.95,
				theta_init=self.initial_solution,
				num_iters=num_iters,
				store_values=self.write_logfile,
				verbose=kwargs['verbose'])

			if self.write_logfile:
				log_counter = 0
				logdir = os.path.join(os.getcwd(),
					'logs')
				os.makedirs(logdir,exist_ok=True)
				filename = os.path.join(logdir,
					f'candidate_selection_log{log_counter}.p')

				while os.path.exists(filename):
					filename = filename.replace(
						f'log{log_counter}',f'log{log_counter+1}')
					log_counter+=1
				with open(filename,'wb') as outfile:
					pickle.dump(res,outfile)
					print(f"Wrote {filename} with candidate selection log info")

			if res['solution_found']:
				print("Found solution to gradient descent")
				candidate_solution = res['candidate_solution']
			else:
				print("No solution found!")
				candidate_solution = 'NSF'

		elif self.optimization_technique == 'barrier_function':
			if self.optimizer in ['Powell','CG','Nelder-Mead','BFGS']:
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

				es = cma.CMAEvolutionStrategy(self.initial_solution, 0.2,
					kwargs['minimizer_options'])
				# es = cma.CMAEvolutionStrategy(np.zeros_like(initial_solution), 0.5,options)
				
				es.optimize(self.candidate_objective)
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
		else:
			raise NotImplementedError(f"Optimization technique: {self.optimization_technique} is not implemented")

		# Reset parse tree base node dicts, 
		# including data and datasize attributes
		for pt in self.parse_trees:
			pt.reset_base_node_dict(reset_data=True)

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
			# Adding regularization term so that large thetas make this less negative
			# and therefore worse 
			result = -1.0*self.primary_objective(self.model,theta,
				self.candidate_dataset)
			if hasattr(self,'reg_coef'):
				reg_term = self.reg_coef*np.linalg.norm(theta)
				result += reg_term
				# print(f"reg_term = {reg_term}")
		# print(self.candidate_dataset.df.episode_index.nunique(),theta,result)
		# print("Primary objective eval is:", result)
		# Prediction of what the safety test will return. 
		# Initialized to pass
		predictSafetyTest = True     
		for tree_i,pt in enumerate(self.parse_trees): 
			# before we propagate, reset the bounds on all base nodes
			pt.reset_base_node_dict()

			bounds_kwargs = dict(
				theta=theta,
				dataset=self.candidate_dataset,
				model=self.model,
				bound_method='ttest',
				branch='candidate_selection',
				n_safety=self.n_safety,
				regime=self.regime)

			if hasattr(self,'scaler'):
				bounds_kwargs['scaler'] = self.scaler

			pt.propagate_bounds(**bounds_kwargs)
			
			# Check if the i-th behavioral constraint is satisfied
			upper_bound = pt.root.upper 
			# print(f"Upper_bound on ghat: {upper_bound}") 
			
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
					if self.optimization_technique == 'barrier_function':
						# print("here")
						result = 100000.0    
				# Add a shaping to the objective function that will 
				# push the search toward solutions that will pass 
				# the prediction of the safety test

				result = result + upper_bound
		
		# print(f"Primary objective: {result}, ghat_upper_bound: {upper_bound}")
		
		# title = f'Parse tree'
		# graph = pt.make_viz(title)
		# graph.attr(fontsize='12')
		# graph.view() # to open it as a pdf
		# input("End of optimzer iteration")
		return result

	def evaluate_primary_objective(self,theta):
		# Get value of the primary objective given model weights
		if self.regime == 'supervised':
			result = self.primary_objective(self.model, theta, 
					self.features.values, self.labels.values)
			return result

		elif self.regime == 'RL':
			print("Evaluating primary objective")
			# Want to maximize the importance weight so minimize negative importance weight
			# Adding regularization term so that large thetas make this less negative
			# and therefore worse 
			result = -1.0*self.primary_objective(self.model,theta,
				self.candidate_dataset)
			if hasattr(self,'reg_coef'):
				reg_term = self.reg_coef*np.linalg.norm(theta)
				result += reg_term
			return result

	def get_constraint_upper_bound(self,theta):
		pt = self.parse_trees[0]
		pt.reset_base_node_dict()

		bounds_kwargs = dict(
			theta=theta,
			dataset=self.candidate_dataset,
			model=self.model,
			bound_method='ttest',
			branch='candidate_selection',
			n_safety=self.n_safety,
			regime=self.regime
			)

		if hasattr(self,'scaler'):
			bounds_kwargs['scaler'] = self.scaler

		pt.propagate_bounds(**bounds_kwargs)

		return pt.root.upper