""" Command line interface for supervised learning Seldonian algorithms

Usage:

    .. code-block:: console

        $ python command_line_interface.py data_pth metadata_pth
"""

import os
import sys
import json
import argparse
import pickle
from seldonian.parse_tree.parse_tree import ParseTree
from seldonian.dataset import DataSetLoader
from seldonian.utils.io_utils import (
	dir_path, yes_or_no_input, save_pickle)

def run_interface(
		data_pth,
		metadata_pth,
		include_sensitive_columns=False,
		include_intercept_term=False,
		save_dir='.'):
	""" Runs the command line interface 

	:param data_pth: Path to main dataset file
	:type data_pth: str

	:param metadata_pth: Path to metadata JSON file
	:type metadadata_pth: str
	"""
	print()
	print("###############################################")
	print("Welcome to the Seldonian Command Line Interface")
	print("###############################################")
	print()
	input("Hit any key to get started: ")
	print()
	# Load metadata
	print("Reading metadata file...")
	with open(args.metadata_pth,'r') as infile:
		metadata_dict = json.load(infile)

	regime = metadata_dict['regime']
	print(f"The regime is: {regime}")
	print()
	columns = metadata_dict['columns']
	dataset_kwargs = dict(
		column_names=columns,
		regime=regime)

	if regime == 'supervised':
		sensitive_columns = metadata_dict['sensitive_columns']
		label_column = metadata_dict['label_column']
		sub_regime = metadata_dict['sub_regime']
		dataset_kwargs['label_column'] = label_column
		dataset_kwargs['sensitive_column_names'] = sensitive_columns

		include_sensitive_columns = False
		if sensitive_columns != []:
			print(f"Identified the following sensitive columns: {sensitive_columns}")
			sensitive_cols_str_to_show = (
				f"Do you want to include these sensitive columns "
		         "as features during training? ")
			include_sensitive_columns = yes_or_no_input(
				sensitive_cols_str_to_show,default_str='n',default_val=False)
		dataset_kwargs['include_sensitive_columns'] = include_sensitive_columns

		intercept_str_to_show = (
			f"Insert ones as first column in feature array")

		include_intercept_term = yes_or_no_input(
			intercept_str_to_show,default_str='n',
			default_val=False)

		dataset_kwargs['include_intercept_term'] = include_intercept_term
	elif regime == 'RL':
		sub_regime = 'all'
		
	loader = DataSetLoader(**dataset_kwargs
		)

	print("Loading data...")
	print()
	ds = loader.from_csv(data_pth)
	print(f"Loaded a dataset with dataframe:")
	print(ds.df)
	# Save dataset object
	print()

	################################
	## Collect constraints from user 
	################################
	n_constraints = int(input("How many constraints do you want to add? "))
	for ii in range(n_constraints):
		constraint_number = ii+1
		constraint_str = input(f"Enter constraint #{constraint_number} (str): ")
		print()
		delta = float(
			input(
				"Enter the value of delta for this constraint (float between 0 and 1): ")
			)
		print()
		# Validate their entry by creating a parse tree from it

		# Instantiate parse tree object
		parse_tree = ParseTree(delta=delta,regime=regime,
			sub_regime=sub_regime)

		# Fill out tree
		parse_tree.create_from_ast(constraint_str)
		# assign deltas for each base node
		# use equal weighting for each base node
		parse_tree.assign_deltas(weight_method='equal')

		# Assign bounds needed on the base nodes
		parse_tree.assign_bounds_needed()

		print("Successfully created parse tree from your constraint.")
		view_parse_tree = yes_or_no_input(
			"Would you like to view the parse tree for this constraint?",
			default_str='n',default_val=False)
		# # Create the graphviz visualization and render it to a PDF
		title = f'Parse tree for expression:\n{constraint_str}\ndelta={delta}'
		graph = parse_tree.make_viz(title)
		graph.attr(fontsize='12')
		if view_parse_tree:
			graph.view() # to open it as a pdf
			input("Hit any key to continue: ")
		# Save graph of parse tree

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# Required args
	parser.add_argument('data_pth',  type=str,
		help='Path to data file')
	parser.add_argument('metadata_pth',  type=str,
		help='Path to metadata file')
	# Optional args
	parser.add_argument('--save_dir',  type=dir_path, default='.',
		help="Folder in which to save interface outputs")
	args = parser.parse_args()
	run_interface(
		args.data_pth,
		args.metadata_pth,
		args.save_dir)
	