import os
import sys
import json
import argparse
import pickle
from seldonian.parse_tree import ParseTree
from seldonian.dataset import DataSetLoader
from seldonian.io_utils import dir_path

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

	# Load metadata
	with open(args.metadata_pth,'r') as infile:
		metadata_dict = json.load(infile)

	regime = metadata_dict['regime']
	columns = metadata_dict['columns']
	sensitive_columns = metadata_dict['sensitive_columns']

	if regime == 'supervised':
		label_column = metadata_dict['label_column']

	# Load dataset from file
	loader = DataSetLoader(
		column_names=columns,
		sensitive_column_names=sensitive_columns,
		include_sensitive_columns=args.include_sensitive_columns,
		include_intercept_term=args.include_intercept_term,
		label_column=label_column,
		regime=regime)

	ds = loader.from_csv(args.data_pth)
	
	# Save dataset object
	ds_save_dir = os.path.join(args.save_dir,'dataset.p')
	with open(ds_save_dir,'wb') as outfile:
		pickle.dump(ds,outfile,protocol=pickle.HIGHEST_PROTOCOL)
		print(f"Saved {ds_save_dir}\n")
	# constraint_str = input("Enter your constraint: ")
	# constraint_str = 'abs((Mean_Error | [M]) - (Mean_Error | [F])) - 0.05'

	# constraint_str = 'X**Y - 2.0'
	# constraint_str = '(Mean_Error | [M]) - 0.1'
	# constraint_str = 'FPR | [M] - FNR | [F] - 0.1'
	# constraint_str = 'F1**3 + FPR**3'
	# constraint_str = 'abs(MED_MF) - 0.05'
	# constraint_str = 'abs(ME | [M])'
	constraint_strs = ['abs((PR | [M]) - (PR | [F])) - 0.15'] 
	constraint_names = ['demographic_parity']
	# constraint_names = ['disparate_impact']
	# constraint_strs = ['0.8 - min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M]))']
	deltas = [0.05]

	# For each constraint, make a parse tree
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
		# Save parse tree

		pt_save_pth = os.path.join(args.save_dir,f'parse_tree_{constraint_name}.p')
		with open(pt_save_pth,'wb') as outfile:
			pickle.dump(parse_tree,outfile,protocol=pickle.HIGHEST_PROTOCOL)
			print(f"Saved {pt_save_pth}")

		# print(parse_tree.root.left.will_lower_bound)
		# print(parse_tree.root.left.will_upper_bound)
		# # # # Propagate bounds using random interval assignment to base variables
		# parse_tree.propagate_bounds(bound_method='manual')

		# # # Create the graphviz visualization and render it to a PDF
		# title = f'Parse tree for expression:\n{constraint_str}\ndelta={delta}'
		# graph = parse_tree.make_viz(title)
		# graph.attr(fontsize='12')
		# graph.view() # to open it as a pdf
