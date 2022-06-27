import os
import sys
import json
import argparse
import pickle
import importlib


from seldonian.utils.io_utils import dir_path
from seldonian.parse_tree.parse_tree import ParseTree
from seldonian.dataset import DataSetLoader
from seldonian.spec import RLSpec
from seldonian.models.model import *


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
	print()

	# Load metadata
	with open(args.metadata_pth,'r') as infile:
		metadata_dict = json.load(infile)

	regime = metadata_dict['regime']
	columns = metadata_dict['columns']
	
	RL_environment_name = metadata_dict['RL_environment_name']
	RL_environment_module = importlib.import_module(
		f'seldonian.RL.environments.{RL_environment_name}')
	RL_environment_obj = RL_environment_module.Environment()	
	
	model_class = TabularSoftmaxModel
	model_instance = model_class(RL_environment_obj)

	primary_objective = model_instance.default_objective 

	# Load dataset from file
	loader = DataSetLoader(
		regime=regime)

	dataset = loader.load_RL_dataset(
		filename=args.data_pth,
		metadata_filename=args.metadata_pth,
		file_type='csv')
	
	constraint_strs = ['-0.25 - J_pi_new'] 
	
	deltas = [0.05]

	# For each constraint, make a parse tree
	parse_trees = []
	for ii in range(len(constraint_strs)):
		constraint_str = constraint_strs[ii]

		delta = deltas[ii]
		# Create parse tree object
		parse_tree = ParseTree(delta=delta,regime='RL',
		sub_regime='all')

		# Fill out tree
		parse_tree.create_from_ast(constraint_str)
		# assign deltas for each base node
		# use equal weighting for each base node
		parse_tree.assign_deltas(weight_method='equal')

		# Assign bounds needed on the base nodes
		parse_tree.assign_bounds_needed()
		
		parse_trees.append(parse_tree)

	# Save spec object, using defaults where necessary
	spec = RLSpec(
		dataset=dataset,
		model_class=model_class,
		frac_data_in_safety=0.6,
		primary_objective=primary_objective,
		parse_trees=parse_trees,
		RL_environment_obj=RL_environment_obj,
		initial_solution_fn=None,
		bound_method='ttest',
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : 0.5,
		    'alpha_theta'   : 0.005,
		    'alpha_lamb'    : 0.005,
		    'beta_velocity' : 0.9,
		    'beta_rmsprop'  : 0.95,
		    'num_iters'     : 20,
		    'gradient_library': "autograd",
		    'hyper_search'  : None,
		    'verbose'       : True,
		},
		regularization_hyperparams={},
		normalize_returns=False,
	)

	spec_save_name = os.path.join(args.save_dir,'spec.pkl')
	with open(spec_save_name,'wb') as outfile:
		pickle.dump(spec,outfile,protocol=pickle.HIGHEST_PROTOCOL)
		print(f"Saved Spec object to: {spec_save_name}\n")
