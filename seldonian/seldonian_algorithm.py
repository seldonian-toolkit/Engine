from sklearn.model_selection import train_test_split

from seldonian.dataset import DataSet
from seldonian.candidate_selection import CandidateSelection
from seldonian.safety_test import SafetyTest

def seldonian_algorithm(spec):
	"""
	Takes a spec object and run seldonian algorithm
	"""

	model_instance = spec.model()
	dataset = spec.dataset
	regime = dataset.regime
	column_names = dataset.df.columns

	candidate_df, safety_df = train_test_split(
			dataset.df, test_size=spec.frac_data_in_safety, shuffle=False)
	
	if regime == 'supervised':
		label_column = dataset.label_column
		include_sensitive_columns = dataset.include_sensitive_columns
		include_intercept_term = dataset.include_intercept_term
		sensitive_column_names = dataset.sensitive_column_names
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

	# Candidate selection
	cs = CandidateSelection(
		model=model_instance,
		candidate_dataset=candidate_dataset,
		n_safety=n_safety,
		parse_trees=spec.parse_trees,
		primary_objective=spec.primary_objective,
		optimization_technique=spec.optimization_technique,
		optimizer=spec.optimizer,
		initial_solution=initial_solution,
		write_logfile=True)

	candidate_solution = cs.run(**spec.optimization_hyperparams)
	print(candidate_solution) # array-like or "NSF"
	
	NSF=False
	if type(candidate_solution) == str and candidate_solution == 'NSF':
		NSF = True
	
	if NSF:
		passed_safety=False
	else:
		# Safety test
		st = SafetyTest(safety_dataset,model_instance,spec.parse_trees)
		passed_safety = st.run(candidate_solution,bound_method='ttest')
		if passed_safety:
			print("Passed safety test")
		else:
			print("Failed safety test")

	return passed_safety, candidate_solution