""" Module for running the Seldonian algorithm """

from sklearn.model_selection import train_test_split
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from seldonian.dataset import (SupervisedDataSet, RLDataSet)
from seldonian.candidate_selection.candidate_selection import CandidateSelection
from seldonian.safety_test.safety_test import SafetyTest

def seldonian_algorithm(spec):
	"""
	Takes a spec object and run seldonian algorithm

	:param spec: Seldonian specification obect
	:type spec: spec.Spec object
	"""

	dataset = spec.dataset
	regime = dataset.regime
	column_names = dataset.df.columns

	if regime == 'supervised':
		model_instance = spec.model_class()
		candidate_df, safety_df = train_test_split(
			dataset.df, test_size=spec.frac_data_in_safety, shuffle=False)

		label_column = dataset.label_column
		include_sensitive_columns = dataset.include_sensitive_columns
		include_intercept_term = dataset.include_intercept_term
		sensitive_column_names = dataset.sensitive_column_names

		# Create candidate and safety datasets
		candidate_dataset = SupervisedDataSet(
			candidate_df,meta_information=column_names,
			sensitive_column_names=sensitive_column_names,
			include_sensitive_columns=include_sensitive_columns,
			include_intercept_term=include_intercept_term,
			label_column=label_column)

		safety_dataset = SupervisedDataSet(
			safety_df,meta_information=column_names,
			sensitive_column_names=sensitive_column_names,
			include_sensitive_columns=include_sensitive_columns,
			include_intercept_term=include_intercept_term,
			label_column=label_column)
	
		n_safety = len(safety_df)

		# Set up initial solution
		initial_solution_fn = spec.initial_solution_fn
		candidate_labels = candidate_dataset.df[label_column]
		candidate_features = candidate_dataset.df.loc[:,
			column_names != label_column]

		if not include_sensitive_columns:
			candidate_features = candidate_features.drop(
				columns=sensitive_column_names)
	
		if include_intercept_term:
			candidate_features.insert(0,'offset',1.0) # inserts a column of 1's

		initial_solution = initial_solution_fn(candidate_features,candidate_labels)
		
		cs_kwargs = dict(
			model=model_instance,
			candidate_dataset=candidate_dataset,
			n_safety=n_safety,
			parse_trees=spec.parse_trees,
			primary_objective=spec.primary_objective,
			optimization_technique=spec.optimization_technique,
			optimizer=spec.optimizer,
			initial_solution=initial_solution,
			regime=regime,)

		st_kwargs = dict(
			dataset=safety_dataset,
			model=model_instance,parse_trees=spec.parse_trees,
			regime=regime,
			)

	elif regime == 'RL':
		RL_environment_obj = spec.RL_environment_obj
		normalize_returns = spec.normalize_returns

		model_instance = spec.model_class(RL_environment_obj)
		df = spec.dataset.df
		
		# Create candidate and safety datasets
		# Cant train_test_split because we need to separate by episode
		episodes = sorted(df.episode_index.unique())
		n_episodes = len(episodes)
		# For candidate take first 1.0-frac_data_in_safety fraction
		# and for safety take remaining
		n_candidate = int(round(n_episodes*(1.0-spec.frac_data_in_safety)))
		candidate_episodes = episodes[0:n_candidate]
		safety_episodes = episodes[n_candidate:]
		
		safety_df = df.copy().loc[
			df['episode_index'].isin(safety_episodes)]
		candidate_df = df.copy().loc[
			df['episode_index'].isin(candidate_episodes)]	

		candidate_dataset = RLDataSet(
			candidate_df,meta_information=df.columns)

		safety_dataset = RLDataSet(
			safety_df,meta_information=df.columns)

		n_safety = safety_df['episode_index'].nunique()
		n_candidate = candidate_df['episode_index'].nunique()
		print(f"Safety dataset has {n_safety} episodes")
		print(f"Candidate dataset has {n_candidate} episodes")
		
		# initial solution
		initial_solution = RL_environment_obj.initial_weights

		cs_kwargs = dict(
			model=model_instance,
			candidate_dataset=candidate_dataset,
			n_safety=n_safety,
			parse_trees=spec.parse_trees,
			primary_objective=spec.primary_objective,
			optimization_technique=spec.optimization_technique,
			optimizer=spec.optimizer,
			initial_solution=initial_solution,
			regime=regime,
			gamma=RL_environment_obj.gamma,
			normalize_returns=normalize_returns
			)

		if normalize_returns:
			cs_kwargs['min_return']=RL_environment_obj.min_return
			cs_kwargs['max_return']=RL_environment_obj.max_return
			
		st_kwargs = dict(
			dataset=safety_dataset,
			model=model_instance,parse_trees=spec.parse_trees,
			gamma=RL_environment_obj.gamma,
			regime=regime,
			normalize_returns=normalize_returns
			)

		if normalize_returns:
			st_kwargs['min_return']=RL_environment_obj.min_return
			st_kwargs['max_return']=RL_environment_obj.max_return
			
	# Candidate selection
	print(cs_kwargs)
	cs = CandidateSelection(**cs_kwargs,**spec.regularization_hyperparams,
		write_logfile=True)

	candidate_solution = cs.run(**spec.optimization_hyperparams,
		use_builtin_primary_gradient_fn=spec.use_builtin_primary_gradient_fn,
		custom_primary_gradient_fn=spec.custom_primary_gradient_fn)
	
	NSF=False
	if type(candidate_solution) == str and candidate_solution == 'NSF':
		NSF = True
	
	if NSF:
		passed_safety=False
	else:
		# Safety test
		st = SafetyTest(**st_kwargs)
		passed_safety = st.run(candidate_solution,bound_method='ttest')
		if passed_safety:
			print("Passed safety test")
		else:
			print("Failed safety test")

	return passed_safety, candidate_solution