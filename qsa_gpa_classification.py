import sys
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from sklearn.model_selection import train_test_split

from seldonian.dataset import DataSetLoader,DataSet
from seldonian.parse_tree import ParseTree
from seldonian.model import LinearClassifierModel,LogisticRegressionModel
from seldonian.candidate_selection import CandidateSelection
from seldonian.safety_test import SafetyTest

if __name__ == '__main__':
	# gpa dataset
	np.random.seed(0)
	csv_file = '../datasets/GPA/data_classification.csv'
	columns = ["M","F","SAT_Physics",
		   "SAT_Biology","SAT_History",
		   "SAT_Second_Language","SAT_Geography",
		   "SAT_Literature","SAT_Portuguese_and_Essay",
		   "SAT_Math","SAT_Chemistry","GPA_class"]

	sensitive_column_names = ['M','F']
	label_column = "GPA_class"
	include_sensitive_columns=False
	include_intercept_term=True
	loader = DataSetLoader(column_names=columns,
		sensitive_column_names=sensitive_column_names,
		include_sensitive_columns=include_sensitive_columns,
		regime='supervised',label_column=label_column)
	dataset = loader.from_csv(csv_file)

	frac_data_in_safety = 0.6

	candidate_df, safety_df = train_test_split(
			dataset.df, test_size=frac_data_in_safety, shuffle=False)
	
	sensitive_candidate_df = candidate_df[sensitive_column_names]
	sensitive_safety_df = safety_df[sensitive_column_names]
	
	candidate_dataset = DataSet(
		candidate_df,meta_information=dataset.df.columns,
		sensitive_column_names=sensitive_column_names,
		include_sensitive_columns=include_sensitive_columns,
		include_intercept_term=include_intercept_term,
		regime='supervised',label_column=label_column)

	safety_dataset = DataSet(
		safety_df,meta_information=dataset.df.columns,
		sensitive_column_names=sensitive_column_names,
		include_sensitive_columns=include_sensitive_columns,
		include_intercept_term=include_intercept_term,
		regime='supervised',label_column=label_column)
	
	n_safety = len(safety_df)

	# Linear regression model
	model_instance = LinearClassifierModel()
	# model_instance = LogisticRegressionModel()

	# Constraints

	# Demographic parity
	constraint_str = 'abs((PR | [M]) - (PR | [F])) - 0.15'

	# Predictive equality 
	# constraint_str = 'abs((FPR | [M]) - (FPR | [F])) - 0.2'

	# Equal opportunity 
	# constraint_str = 'abs((FNR | [M]) - (FNR | [F])) - 0.2'

	# Equalized odds - predictive equality and equal opportunity - 

	# Disparate impact 
	# constraint_str = '0.8 - min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M]))'

	# Treatment equality
	# constraint_str = '(FNR | [M])/(FPR | [M]) - (FNR | [F])/(FPR | [F]) - 0.2'

	delta = 0.05
	# Create parse tree object
	pt = ParseTree(delta=delta)
	
	pt.create_from_ast(constraint_str)
	
	# assign deltas for each base node
	# use equal weighting for each base node
	pt.assign_deltas(weight_method='equal')
	parse_trees = [pt]
	# title = f'Parse tree for expression:\n{constraint_str}\ndelta={delta}'
	# graph = pt.make_viz(title)
	# graph.attr(fontsize='12')
	# graph.view() # to open it as a pdf
	# input("End of optimzer iteration")
	# Candidate selection
	minimizer_options = {}

	# Set up initial solution
	labels = candidate_dataset.df[label_column]
	features = candidate_dataset.df.loc[:,
		candidate_dataset.df.columns != label_column]
	if not include_sensitive_columns:
		features = features.drop(
			columns=candidate_dataset.sensitive_column_names)
	if include_intercept_term:
		features.insert(0,'offset',1.0) # inserts a column of 1's
	initial_solution = model_instance.fit(features,labels)

	cs = CandidateSelection(
		model=model_instance,
		candidate_dataset=candidate_dataset,
		n_safety=n_safety,
		parse_trees=parse_trees,
		primary_objective=model_instance.perceptron_loss,
		optimization_technique='barrier_function',
		optimizer='Nelder-Mead',
		# optimizer='CMA-ES',
		initial_solution=initial_solution)

	candidate_solution = cs.run(minimizer_options=minimizer_options)
	print(candidate_solution)
	
	# # Safety test
	st = SafetyTest(safety_dataset,model_instance,parse_trees)
	passed_safety = st.run(candidate_solution,bound_method='ttest')
	print(passed_safety)

	# # title = f'Parse tree for expression:\n{constraint_str}\ndelta={delta}'
	# # graph = pt.make_viz(title)
	# # graph.attr(fontsize='12')
	# # graph.view() # to open it as a pdf

	# # Estimate actual abs(FPR|M - FPR|F) on the safety set
	# safety_df.insert(0,'offset',1.0) # inserts a column of 1's
	# safety_df_M = safety_df.loc[sensitive_safety_df['M']==1]
	# safety_features_M = safety_df_M.drop(columns=['M','F','GPA_class'])
	# safety_labels_M = safety_df_M['GPA_class']

	# safety_df_F = safety_df.loc[sensitive_safety_df['F']==1]
	# safety_features_F = safety_df_F.drop(columns=['M','F','GPA_class'])
	# safety_labels_F = safety_df_F['GPA_class']
	# safety_FPR_M = model_instance.sample_False_Positive_Rate(
	# 	model_instance,candidate_solution,safety_features_M,safety_labels_M)
	# safety_FPR_F = model_instance.sample_False_Positive_Rate(
	# 	model_instance,candidate_solution,safety_features_F,safety_labels_F)
	# print(safety_FPR_M,safety_FPR_F,abs(safety_FPR_M-safety_FPR_F))

	