""" Module for running Seldonian algorithms """
import copy

from sklearn.model_selection import train_test_split
import autograd.numpy as np   # Thinly-wrapped version of Numpy

import warnings
from seldonian.warnings.custom_warnings import *
from seldonian.dataset import (SupervisedDataSet, RLDataSet)
from seldonian.candidate_selection.candidate_selection import CandidateSelection
from seldonian.safety_test.safety_test import SafetyTest

class SeldonianAlgorithm():
	def __init__(self,spec):
		""" Object for running the Seldonian algorithm and getting 
		introspection into candidate selection and safety test 


		:param spec: The specification object with the complete 
			set of parameters for running the Seldonian algorithm
		:type spec: :py:class:`.Spec` object
		"""
		self.spec = spec
		self.parse_trees = self.spec.parse_trees
		self.base_node_bound_method_dict = self.spec.base_node_bound_method_dict

		if self.base_node_bound_method_dict != {}:
			all_pt_constraint_strs = [pt.constraint_str for pt in self.parse_trees]
			for constraint_str in self.base_node_bound_method_dict:
				this_bound_method_dict = self.base_node_bound_method_dict[constraint_str]
				# figure out which parse tree this comes from
				this_pt_index = all_pt_constraint_strs.index(constraint_str)
				this_pt = self.parse_trees[this_pt_index]
				# change the bound method for each node provided
				for node_name in this_bound_method_dict:
					this_pt.base_node_dict[node_name]['bound_method'] = this_bound_method_dict[node_name]

		self.dataset = self.spec.dataset
		self.regime = self.dataset.regime
		self.column_names = self.dataset.meta_information

		if self.regime == 'supervised':
			self.model_instance = self.spec.model_class()
			self.candidate_df, self.safety_df = train_test_split(
				self.dataset.df, test_size=self.spec.frac_data_in_safety, 
				shuffle=False)

			self.label_column = self.dataset.label_column
			self.include_sensitive_columns = self.dataset.include_sensitive_columns
			self.include_intercept_term = self.dataset.include_intercept_term
			self.sensitive_column_names = self.dataset.sensitive_column_names

			# Create candidate and safety datasets
			self.candidate_dataset = SupervisedDataSet(
				self.candidate_df,meta_information=self.column_names,
				sensitive_column_names=self.sensitive_column_names,
				include_sensitive_columns=self.include_sensitive_columns,
				include_intercept_term=self.include_intercept_term,
				label_column=self.label_column)

			self.safety_dataset = SupervisedDataSet(
				self.safety_df,meta_information=self.column_names,
				sensitive_column_names=self.sensitive_column_names,
				include_sensitive_columns=self.include_sensitive_columns,
				include_intercept_term=self.include_intercept_term,
				label_column=self.label_column)
			
			self.n_candidate = len(self.candidate_df)
			self.n_safety = len(self.safety_df)

			if self.n_candidate < 2 or self.n_safety < 2:
				warning_msg = (
					"Warning: not enough data to "
					"run the Seldonian algorithm.")
				warnings.warn(warning_msg)

			# Set up initial solution
			self.initial_solution_fn = self.spec.initial_solution_fn

			self.candidate_labels = self.candidate_df[self.label_column]
			self.candidate_features = self.candidate_df.loc[:,
				self.candidate_df.columns != self.label_column]

			if not self.include_sensitive_columns:
				self.candidate_features = self.candidate_features.drop(
					columns=self.sensitive_column_names)
		
			if self.include_intercept_term:
				self.candidate_features.insert(0,'offset',1.0) # inserts a column of 1's

			try: 
				self.initial_solution = self.initial_solution_fn(
					self.candidate_features,self.candidate_labels)
			except Exception as e: 
				warning_msg = (
					"Warning: initial solution function failed with this error:"
					f" {e}")
				warnings.warn(warning_msg)
				self.initial_solution = np.random.normal(
					loc=0.0,scale=1.0,size=(self.candidate_features.shape[1]+1)
					)
			print("Initial solution: ")
			print(self.initial_solution)

		elif self.regime == 'RL':
			self.RL_environment_obj = self.spec.RL_environment_obj
			self.normalize_returns = self.spec.normalize_returns

			self.RL_agent_obj = self.spec.RL_agent_obj

			self.model_instance = self.spec.model_class(
				self.RL_agent_obj, self.RL_environment_obj)

			episodes = self.spec.dataset.episodes
			# Create candidate and safety datasets
			n_episodes = len(episodes)
			# For candidate take first 1.0-frac_data_in_safety fraction
			# and for safety take remaining
			self.n_candidate = int(round(n_episodes*(1.0-self.spec.frac_data_in_safety)))
			self.n_safety = n_episodes - self.n_candidate
			candidate_episodes = episodes[0:self.n_candidate]
			safety_episodes = episodes[self.n_candidate:]

			self.candidate_dataset = RLDataSet(
				episodes=candidate_episodes,
				meta_information=self.column_names)

			self.safety_dataset = RLDataSet(
				episodes=safety_episodes,
				meta_information=self.column_names)
			# assert len(safety_df) == n_safety
			print(f"Safety dataset has {self.n_safety} episodes")
			print(f"Candidate dataset has {self.n_candidate} episodes")
			
			# initial solution
			self.initial_solution = self.RL_agent_obj.get_params()

			cs_kwargs = dict(
				model=self.model_instance,
				candidate_dataset=self.candidate_dataset,
				n_safety=self.n_safety,
				parse_trees=self.spec.parse_trees,
				primary_objective=self.spec.primary_objective,
				optimization_technique=self.spec.optimization_technique,
				optimizer=self.spec.optimizer,
				initial_solution=self.initial_solution,
				regime=self.regime,
				gamma=self.RL_environment_obj.gamma,
				normalize_returns=self.normalize_returns
				)

			if self.normalize_returns:
				cs_kwargs['min_return']=self.RL_environment_obj.min_return
				cs_kwargs['max_return']=self.RL_environment_obj.max_return
			
	def candidate_selection(self,write_cs_logfile=False):
		""" Creat the candidate selection object """
		if self.regime == 'supervised':
			cs_kwargs = dict(
				model=self.model_instance,
				candidate_dataset=self.candidate_dataset,
				n_safety=self.n_safety,
				parse_trees=self.spec.parse_trees,
				primary_objective=self.spec.primary_objective,
				optimization_technique=self.spec.optimization_technique,
				optimizer=self.spec.optimizer,
				initial_solution=self.initial_solution,
				regime=self.regime)

		elif self.regime == 'RL':
			cs_kwargs = dict(
				model=self.model_instance,
				candidate_dataset=self.candidate_dataset,
				n_safety=self.n_safety,
				parse_trees=self.spec.parse_trees,
				primary_objective=self.spec.primary_objective,
				optimization_technique=self.spec.optimization_technique,
				optimizer=self.spec.optimizer,
				initial_solution=self.initial_solution,
				regime=self.regime,
				gamma=self.RL_environment_obj.gamma,
				normalize_returns=self.normalize_returns
				)

			if self.normalize_returns:
				cs_kwargs['min_return']=self.RL_environment_obj.min_return
				cs_kwargs['max_return']=self.RL_environment_obj.max_return

		cs = CandidateSelection(**cs_kwargs,**self.spec.regularization_hyperparams,
			write_logfile=write_cs_logfile)

		return cs

	def safety_test(self):
		""" Create the safety test object """
		if self.regime == 'supervised':
			st_kwargs = dict(
				safety_dataset=self.safety_dataset,
				model=self.model_instance,parse_trees=self.spec.parse_trees,
				regime=self.regime,
				)	
		elif self.regime == 'RL':
			st_kwargs = dict(
				safety_dataset=self.safety_dataset,
				model=self.model_instance,parse_trees=self.spec.parse_trees,
				gamma=self.RL_environment_obj.gamma,
				regime=self.regime,
				normalize_returns=self.normalize_returns
				)

			if self.normalize_returns:
				st_kwargs['min_return']=self.RL_environment_obj.min_return
				st_kwargs['max_return']=self.RL_environment_obj.max_return
		
		st = SafetyTest(**st_kwargs)
		return st

	def run(self,write_cs_logfile=False):
		"""
		Runs seldonian algorithm using spec object

		:param write_cs_logfile: Whether to write out the pickle file 
			containing the evolution of the parameters of candidate selection

		:return: (passed_safety, solution). passed_safety 
			indicates whether solution found during candidate selection
			passes the safety test. solution is the optimized
			model weights found during candidate selection or 'NSF'.
		:rtype: Tuple 
		"""
			
		cs = self.candidate_selection(write_cs_logfile=write_cs_logfile)
		solution = cs.run(**self.spec.optimization_hyperparams,
			use_builtin_primary_gradient_fn=self.spec.use_builtin_primary_gradient_fn,
			custom_primary_gradient_fn=self.spec.custom_primary_gradient_fn)
		
		NSF=False
		if type(solution) == str and solution == 'NSF':
			NSF = True

		if NSF:
			passed_safety=False
		else:
			# Safety test
			st = self.safety_test()
			passed_safety = st.run(solution)
		
		return passed_safety, solution

	def evaluate_primary_objective(self,branch,theta):
		""" Get value of the primary objective given model weights,
		theta, on either the candidate selection dataset 
		or the safety dataset. This is just a wrapper for
		primary_objective where data is fixed.

		:param branch: 'candidate_selection' or 'safety_test'
		:type branch: str

		:param theta: model weights
		:type theta: numpy.ndarray

		:return: result, the value of the primary objective 
			evaluated for the given branch at the provided
			value of theta
		:rtype: float
		"""
		
		if branch == 'safety_test':
			st = self.safety_test()
			result = st.evaluate_primary_objective(theta,
				self.spec.primary_objective)
			
		elif branch == 'candidate_selection':
			cs = self.candidate_selection()
			result = cs.evaluate_primary_objective(theta)
		return result