import sys
import numpy as np
from sklearn.model_selection import train_test_split

from src.dataset import DataSetLoader,DataSet
from src.parse_tree import ParseTree
from src.candidate_selection import CandidateSelection
from src.safety_test import SafetyTest


if __name__ == '__main__':
	# gpa dataset
	np.random.seed(0)
	csv_file = '../datasets/GPA/data_phil_modified.csv'
	columns = ["M","F","SAT_Physics",
		   "SAT_Biology","SAT_History",
		   "SAT_Second_Language","SAT_Geography",
		   "SAT_Literature","SAT_Portuguese_and_Essay",
		   "SAT_Math","SAT_Chemistry","GPA"]
	sensitive_column_names = ['M','F']
	label_column = "GPA"
	loader = DataSetLoader(column_names=columns,
		sensitive_column_names=sensitive_column_names,
		regime='supervised',label_column=label_column)
	dataset = loader.from_csv(csv_file)
	# sensitive_df = dataset.df[sensitive_column_names]
	
	frac_data_in_safety = 0.6
	# (sensitive_candidate_df,
	# 	sensitive_safety_df) = train_test_split(sensitive_df,
	# 	test_size=frac_data_in_safety,shuffle=False)

	candidate_df, safety_df = train_test_split(
			dataset.df, test_size=frac_data_in_safety, shuffle=False)
	
	sensitive_candidate_df = candidate_df[sensitive_column_names]
	sensitive_safety_df = safety_df[sensitive_column_names]
	

	candidate_dataset = DataSet(
		candidate_df,meta_information=dataset.df.columns,
		sensitive_column_names=sensitive_column_names,
		regime='supervised',label_column=label_column)

	safety_dataset = DataSet(
		safety_df,meta_information=dataset.df.columns,
		sensitive_column_names=sensitive_column_names,
		regime='supervised',label_column=label_column)
	
	n_safety = len(safety_df)
	print(n_safety)
	# Linear regression model
	from src.model import LRModel
	model_instance = LRModel()

	# Constraints
	constraint_str = 'abs((Mean_Error | [M]) - (Mean_Error | [F])) - 0.05'
	delta = 0.05
	# Create parse tree object
	pt = ParseTree(delta=delta)
	
	pt.create_from_ast(constraint_str)
	
	# assign deltas for each base node
	# use equal weighting for each base node
	pt.assign_deltas(weight_method='equal')
	parse_trees = [pt]
	
	# # Candidate selection
	minimizer_options = {}

	cs = CandidateSelection(
		model=model_instance,
		candidate_dataset=candidate_dataset,
		n_safety=n_safety,
		parse_trees=parse_trees,
		primary_objective=model_instance.sample_Mean_Squared_Error,
		optimization_technique='barrier_function',
		# optimizer='Powell',
		optimizer='cmaes',
		initial_solution_fn=model_instance.fit)

	# try Phil's solution that worked for him -- put last value first since it is offset term
	# candidate_solution = np.array([0.405349,-0.00129556,0.00134197,-0.000391698,0.000718466,-0.00100044,0.0038478,0.00260554,-0.00189149,0.000285684])
	candidate_solution = np.array([0.405349, -0.00129964,   0.00133977, -0.000394042,  0.000716346,  -0.00100369,   0.00384656,   0.00260374,  -0.00189604,  0.000282313])
	# theta = np.array([0.405349,-0.0012894,0.0013513,-0.000405117,0.000711782,-0.00100753,0.00385943,0.00261718 ,-0.00188643,0.000302202])
	# theta = theta[::-1]
	# pt.propagate_bounds(
	# 		theta=theta,
	# 		dataset=candidate_dataset,
	# 		model=model_instance,
	# 		bound_method='ttest',
	# 		branch='candidate_selection',
	# 		n_safety=n_safety)
	
	# safety_labels = safety_dataset.df[label_column]
	# safety_features = safety_dataset.df.loc[:,
	# 	safety_dataset.df.columns != label_column]
	# safety_features.insert(0,'offset',1.0) # inserts a column of 1's
	# safety_features = safety_features.drop(columns=sensitive_column_names)
	# print(safety_features)
	# print(theta.shape)
	# print(safety_features.shape)
	# print(model_instance.sample_Mean_Squared_Error(theta,safety_features,safety_labels))
	# title = f'Parse tree for expression:\n{constraint_str}\ndelta={delta}'
	# graph = pt.make_viz(title)
	# graph.attr(fontsize='12')
	# graph.view() # to open it as a pdf
	# input("Next iteration")
	# candidate_solution = cs.run(minimizer_options=minimizer_options)
	# print("Obtained candidate solution:",candidate_solution)

	# Safety test
	st = SafetyTest(safety_dataset,model_instance,parse_trees)
	passed_safety = st.run(candidate_solution,bound_method='ttest')
	print(passed_safety)
	# title = f'Parse tree for expression:\n{constraint_str}\ndelta={delta}'
	# graph = pt.make_viz(title)
	# graph.attr(fontsize='12')
	# graph.view() # to open it as a pdf
	# if passed_safety:
	# 	print("Passed safety test")
	# 	safety_labels = safety_dataset.df[label_column]
	# 	safety_features = safety_dataset.df.loc[:,
	# 		safety_dataset.df.columns != label_column]
	# 	safety_features.insert(0,'offset',1.0) # inserts a column of 1's
	# 	safety_features = safety_features.drop(columns=sensitive_column_names)
	# 	pri_obj = model_instance.sample_Mean_Squared_Error(
	# 		candidate_solution, safety_features, safety_labels)
	# 	print(f"Primary objective of solution (computed on safety data, D): {pri_obj:g}") 

	# 	safety_features_male = safety_features.loc[sensitive_safety_df['M'] == 1]
	# 	safety_labels_male = safety_labels.loc[sensitive_safety_df['M'] == 1]
	# 	safety_features_female = safety_features.loc[sensitive_safety_df['F'] == 1]
	# 	safety_labels_female = safety_labels.loc[sensitive_safety_df['F'] == 1]

	# 	mean_error_male = model_instance.sample_Mean_Error(
	# 		candidate_solution,safety_features_male,safety_labels_male)
	# 	mean_error_female = model_instance.sample_Mean_Error(
	# 		candidate_solution,safety_features_female,safety_labels_female)
	# 	print(mean_error_male)
	# 	print(mean_error_female)
	# 	absdifference = abs(mean_error_male-mean_error_female)
	# 	print(f"abs((Mean_Error | M) - (Mean_Error | F) ): {absdifference:g}")
	# else:
	# 	print("Failed safety test")
	