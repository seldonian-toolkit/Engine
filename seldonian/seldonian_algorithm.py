""" Module for running the Seldonian algorithm """

from sklearn.model_selection import train_test_split
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from seldonian.dataset import DataSet
from seldonian.candidate_selection import CandidateSelection
from seldonian.safety_test import SafetyTest

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
		candidate_dataset = DataSet(
			candidate_df,meta_information=column_names,
			sensitive_column_names=sensitive_column_names,
			include_sensitive_columns=include_sensitive_columns,
			include_intercept_term=include_intercept_term,
			regime='supervised',label_column=label_column)

		safety_dataset = DataSet(
			safety_df,meta_information=column_names,
			sensitive_column_names=sensitive_column_names,
			include_sensitive_columns=include_sensitive_columns,
			include_intercept_term=include_intercept_term,
			regime='supervised',label_column=label_column)
	
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

		candidate_dataset = DataSet(
			candidate_df,meta_information=df.columns,
			regime=regime)

		safety_dataset = DataSet(
			safety_df,meta_information=df.columns,
			regime=regime)

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
	cs = CandidateSelection(**cs_kwargs,
		write_logfile=True)

	candidate_solution = cs.run(**spec.optimization_hyperparams,
		**spec.regularization_hyperparams)
	# candidate_solution = np.array([ 1.57886344, -1.57224782,  1.58788506,  1.55295861,  1.58501672, -1.58765016,
 # -1.590077,    1.58926174,  1.56557285, -1.55049646, -1.61179002,  1.61905064,
 #  1.578605,   -1.57800341,  1.56918053, -1.54473231,  1.57883498, -1.57466679,
 # -1.56365391,  1.57781652,  1.48187932,  1.64122164, -1.59603775,  1.61112981,
 #  1.58580154,  1.56346047,  1.624256,   -1.57061631, -1.58188818, -1.58334667,
 # -1.54778041,  1.56841265])
	print("candidate solution:")
	print(candidate_solution) # array-like or "NSF"

	
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