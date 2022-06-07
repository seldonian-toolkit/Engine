import os
import sys
import json
import argparse
import pickle
from seldonian.parse_tree.parse_tree import ParseTree
from seldonian.dataset import DataSetLoader
from seldonian.utils.io_utils import dir_path
from seldonian.spec import Spec
from seldonian.models.model import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# Required args
	parser.add_argument('data_pth',  type=str,
		help='Path to data file')
	parser.add_argument('metadata_pth',  type=str,
		help='Path to metadata file')
	
	# Optional args
	parser.add_argument('--include_sensitive_columns',  action='store_true',
		help="Whether to include sensitive columns as predictive features")
	parser.add_argument('--include_intercept_term',  action='store_true',
		help="Whether to add columns of ones in leftmost column")
	parser.add_argument('--save_dir',  type=dir_path, default='.',
		help="Folder in which to save interface outputs")

	args = parser.parse_args()
	print()

	# Load metadata
	with open(args.metadata_pth,'r') as infile:
		metadata_dict = json.load(infile)

	regime = metadata_dict['regime']
	columns = metadata_dict['columns']
	sensitive_columns = metadata_dict['sensitive_columns']

	if regime == 'supervised':
		sub_regime = metadata_dict['sub_regime']
		label_column = metadata_dict['label_column']
		
		# Default model for supervised learning
		if sub_regime == 'classification':
			model_class = LogisticRegressionModel
		elif sub_regime == 'regression':
			model_class = LinearRegressionModel
		else:
			raise NotImplementedError(f"{sub_regime} is not a supported "
				"sub regime of supervised learning")
	elif regime == 'RL':
		# Default model for RL
		model_class = LinearSoftmaxModel

	primary_objective = model_class().default_objective

	# Load dataset from file
	loader = DataSetLoader(
		column_names=columns,
		sensitive_column_names=sensitive_columns,
		include_sensitive_columns=args.include_sensitive_columns,
		include_intercept_term=args.include_intercept_term,
		label_column=label_column,
		regime=regime)

	dataset = loader.from_csv(args.data_pth)
	
	constraint_strs = ['abs((PR | [M]) - (PR | [F])) - 0.15'] 
	constraint_names = ['demographic_parity']
	
	deltas = [0.05]

	# For each constraint, make a parse tree
	parse_trees = []
	for ii in range(len(constraint_strs)):
		constraint_str = constraint_strs[ii]
		constraint_name = constraint_names[ii]

		delta = deltas[ii]
		# Create parse tree object
		parse_tree = ParseTree(delta=delta)

		# Fill out tree
		parse_tree.create_from_ast(constraint_str)
		# assign deltas for each base node
		# use equal weighting for each base node
		parse_tree.assign_deltas(weight_method='equal')

		# Assign bounds needed on the base nodes
		parse_tree.assign_bounds_needed()
		
		parse_trees.append(parse_tree)

	# Save spec object, using defaults where necessary
	spec = Spec(
		dataset=dataset,
		model_class=model_class,
		frac_data_in_safety=0.6,
		primary_objective=primary_objective,
		parse_trees=parse_trees,
		initial_solution_fn=model_class().fit,
		bound_method='ttest',
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : 0.5,
		    'alpha_theta'   : 0.005,
		    'alpha_lamb'    : 0.005,
		    'beta_velocity' : 0.9,
		    'beta_rmsprop'  : 0.95,
		    'num_iters'     : 200,
		    'hyper_search'  : None,
		    'verbose'       : True,
		}
	)

	spec_save_name = os.path.join(args.save_dir,'spec.pkl')
	with open(spec_save_name,'wb') as outfile:
		pickle.dump(spec,outfile,protocol=pickle.HIGHEST_PROTOCOL)
		print(f"Saved Spec object to: {spec_save_name}\n")
