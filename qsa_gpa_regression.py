import autograd.numpy as np   # Thinly-wrapped version of Numpy
from sklearn.model_selection import train_test_split

from seldonian.dataset import *
from seldonian.candidate_selection import CandidateSelection
from seldonian.safety_test import SafetyTest
from seldonian.parse_tree import ParseTree
from seldonian.model import LinearRegressionModel


if __name__ == '__main__':
	print()
	np.random.seed(0)
	csv_file = '../datasets/GPA/data_phil_modified.csv'
	columns = ["M","F","SAT_Physics",
		   "SAT_Biology","SAT_History",
		   "SAT_Second_Language","SAT_Geography",
		   "SAT_Literature","SAT_Portuguese_and_Essay",
		   "SAT_Math","SAT_Chemistry","GPA"]
	sensitive_column_names = ['M','F']
	label_column = 'GPA'
	loader = DataSetLoader(column_names=columns,
		sensitive_column_names=sensitive_column_names,
		regime='supervised',label_column=label_column)
	dataset = loader.from_csv(csv_file)

	candidate_df, safety_df = train_test_split(
			dataset.df, test_size=0.8, shuffle=False)

	candidate_dataset = DataSet(
		candidate_df,meta_information=dataset.df.columns,
		sensitive_column_names=sensitive_column_names,
		regime='supervised',label_column=label_column)

	safety_dataset = DataSet(
		safety_df,meta_information=dataset.df.columns,
		sensitive_column_names=sensitive_column_names,
		regime='supervised',label_column=label_column)
	
	n_safety = len(safety_df)

	# Linear regression model
	model_instance = LinearRegressionModel()

	# Constraints
	constraint_str1 = 'abs(MED_MF) - 0.05'
	delta1 = 0.05

	# Create parse tree object
	pt1 = ParseTree(delta=delta1)

	# Fill out tree
	pt1.create_from_ast(s=constraint_str1)

	# assign delta to single node
	pt1.assign_deltas(weight_method='equal')

	# assign needed bounds
	pt1.assign_bounds_needed()

	# constraint_str2 = '0.0 - MED_MF - 0.05'
	# delta2 = 0.025

	# # Create parse tree object
	# pt2 = ParseTree(delta=delta2)

	# # Fill out tree
	# pt2.create_from_ast(s=constraint_str2)

	# # # assign delta to single node
	# pt2.assign_deltas(weight_method='equal')

	# # assign needed bounds
	# pt2.assign_bounds_needed()

	# title = f'Parse tree for expression:\n{constraint_str1}\ndelta={delta}'
	# graph = pt1.make_viz(title)
	# graph.attr(fontsize='12')
	# graph.view() # to open it as a pdf
	# parse_trees = [pt1,pt2]
	parse_trees = [pt1]

	minimizer_options = {}
	labels = candidate_dataset.df[label_column]
	features = candidate_dataset.df.loc[:,
		candidate_dataset.df.columns != label_column]
	features = features.drop(
		columns=candidate_dataset.sensitive_column_names)
	features.insert(0,'offset',1.0) # inserts a column of 1's
	initial_solution = model_instance.fit(features,labels)

	cs = CandidateSelection(
	    model=model_instance,
	    candidate_dataset=candidate_dataset,
	    n_safety=n_safety,
	    parse_trees=parse_trees,
	    primary_objective=model_instance.sample_Mean_Squared_Error,
	    optimization_technique='barrier_function',
	    optimizer='Nelder-Mead',
	    initial_solution=initial_solution)

	candidate_solution = cs.run(
		minimizer_options=minimizer_options,verbose=True)
	# candidate_solution = np.array([ 2.35142471e+00, -5.92682323e-03,  5.84554876e-04,  3.45285398e-04,
	#   1.89794363e-04,  5.09917663e-05,  3.30307314e-03,  1.56678156e-03,
	#   3.55345048e-04,  3.77028366e-04])
	# # candidate_solution = np.array([ 2.15927631e+00, -6.18007353e-03,  8.92258627e-04,  3.45953630e-04,
	# # 	1.89847929e-04,  5.09917809e-05,  3.43067832e-03,  1.65542813e-03,
	# # 	4.19503821e-04,  3.74452590e-04])
	# candidate_solution = np.array([0.405349,-0.00129556,   0.00134197, -0.000391698,  0.000718466,  -0.00100044,    0.0038478,   0.00260554,  -0.00189149,  0.000285684])
	# candidate_solution = np.array([0.405349,  -0.0012894,    0.0013513, -0.000405117,  0.000711782,  -0.00100753,   0.00385943,   0.00261718,  -0.00188643,  0.000302202])
	# print(candidate_solution)
	st = SafetyTest(safety_dataset,model_instance,parse_trees)
	passed = st.run(candidate_solution,bound_method='ttest')
	print(passed)
	# pt1.propagate_bounds(bound_method='ttest',
	# 	branch='candidate_selection')
	# # Propagate bounds using random interval assignment to base variables
	# parse_tree.propagate_bounds(bound_method='random')

	# # Create the graphviz visualization and render it to a PDF
	# title = f'Parse tree for expression:\n{constraint_str1}\ndelta={delta}'
	# graph = pt1.make_viz(title)
	# graph.attr(fontsize='12')
	# graph.view() # to open it as a pdf
