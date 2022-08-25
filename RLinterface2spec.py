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
from seldonian.models.models import *
from seldonian.RL.environments import *
from seldonian.RL.RL_model import *
from seldonian.models import objectives

def main_RLinterface2spec():
    save_dir, metadata_pth, data_pth = get_paths_from_args()
    dataset = load_dataset(data_pth, metadata_pth)
    dataset2spec(save_dir, metadata_pth, dataset)


def dataset2spec(save_dir, metadata_pth, dataset, policy, constraint_strs):
    # Load metadata
    with open(metadata_pth, 'r') as infile:
        metadata_dict = json.load(infile)

    RL_module_name = metadata_dict['RL_module_name']
    RL_environment_module = importlib.import_module(
        f'seldonian.RL.environments.{RL_module_name}')
    RL_env_class_name = metadata_dict['RL_class_name']
    RL_environment_obj = getattr(RL_environment_module, RL_env_class_name)()

    RL_model_instance = RL_model(policy,RL_environment_obj)
    primary_objective = objectives.IS_estimate

    deltas = [0.05]

    # For each constraint, make a parse tree
    parse_trees = []
    for ii in range(len(constraint_strs)):
        constraint_str = constraint_strs[ii]

        delta = deltas[ii]
        # Create parse tree object
        parse_tree = ParseTree(delta=delta, regime='reinforcement_learning',
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
        model_class=RL_model,
        frac_data_in_safety=0.6,
        primary_objective=primary_objective,
        parse_trees=parse_trees,
        RL_environment_obj=RL_environment_obj,
        RL_policy_obj=policy,
        initial_solution_fn=None,
        optimization_technique='gradient_descent',
        optimizer='adam',
        optimization_hyperparams={
            'lambda_init': 0.5,
            'alpha_theta': 0.005,
            'alpha_lamb': 0.005,
            'beta_velocity': 0.9,
            'beta_rmsprop': 0.95,
            'num_iters': 20,
            'hyper_search': None,
            'gradient_library': 'autograd',
            'verbose': True,
        },
        regularization_hyperparams={},
        normalize_returns=False,
    )

    spec_save_name = os.path.join(save_dir, 'spec.pkl')
    with open(spec_save_name, 'wb') as outfile:
        pickle.dump(spec, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved Spec object to: {spec_save_name}\n")


def get_paths_from_args():
    parser = argparse.ArgumentParser()
    # Required args
    parser.add_argument('data_pth', type=str,
                        help='Path to data file')
    parser.add_argument('metadata_pth', type=str,
                        help='Path to metadata file')

    # Optional args
    parser.add_argument('--save_dir', type=dir_path, default='.',
                        help="Folder in which to save interface outputs")

    args = parser.parse_args()
    print()

    return args.save_dir, args.metadata_pth, args.data_pth


def load_dataset(data_pth, metadata_pth):
    # Load metadata
    with open(metadata_pth, 'r') as infile:
        metadata_dict = json.load(infile)

    # Load dataset from file
    regime = metadata_dict['regime']
    loader = DataSetLoader(
        regime=regime)

    return loader.load_RL_dataset(
        filename=data_pth,
        metadata_filename=metadata_pth,
        file_type='csv')


if __name__ == '__main__':
    main_RLinterface2spec()
