from seldonian.RL.environments.gridworld3x3 import Environment
from time import time
from seldonian.dataset import RLDataSet
from RLinterface2spec import dataset2spec
from seldonian.RL.hyperparams_and_settings import *
from seldonian.RL.RL_runner import run_trial
from seldonian.utils.stats_utils import *
from utils import *
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
from seldonian.RL.Agents import *
from seldonian.RL.environments import *
from seldonian.RL.RL_model import *


def main():

    hyperparameter_and_setting_dict = define_hyperparameter_and_setting_dict()
    start_time = time()
    episodes, agent = run_trial(hyperparameter_and_setting_dict)
    print(f"data generation took {time() - start_time} seconds")

    hyperparameter_and_setting_dict["num_episodes"] = 1
    dataset = RLDataSet(episodes=episodes,meta_information=['O','A','R','pi'])

    # print(dataset.episodes[0])
    # print(f"{len(episodes)} episodes")
    metadata_pth = get_metadata_path(hyperparameter_and_setting_dict["env"])
    save_dir = '.'
    constraint_string = get_constraint_string(hyperparameter_and_setting_dict["env"])
    RL_model = fake_dataset2spec(save_dir, metadata_pth, dataset, agent, constraint_string)
    data_dict = {"episodes" : episodes}
    start_time_is_estimates = time()
    for _ in range(100000):
        mock_vector_IS_estimate(agent.get_params(), data_dict, RL_model)
    print(f"IS estimates took {time() - start_time_is_estimates}")
    print(f"whole thing took {time() - start_time}")


def get_metadata_path(env_name):
    if env_name == "gridworld":
        return "../../static/datasets/RL/gridworld/gridworld_metadata.json"
    elif env_name == "n_step_mountaincar":
        return "../../static/datasets/RL/mountaincar/n_step_mountaincar_metadata.json"
    else:
        error(f"unknown env name {env_name}")

def get_constraint_string(env):
    if env == "gridworld":
        return ['-0.25 - J_pi_new']
    else:
        error(f"Unknown env {env}")

def fake_dataset2spec(save_dir, metadata_pth, dataset, agent, constraint_strs):
    # Load metadata
    with open(metadata_pth, 'r') as infile:
        metadata_dict = json.load(infile)

    RL_module_name = metadata_dict['RL_module_name']
    RL_environment_module = importlib.import_module(
        f'seldonian.RL.environments.{RL_module_name}')
    RL_env_class_name = metadata_dict['RL_class_name']
    RL_environment_obj = getattr(RL_environment_module, RL_env_class_name)()

    RL_model_instance = RL_model(agent,RL_environment_obj)
    primary_objective = RL_model_instance.sample_IS_estimate
    return RL_model_instance


def mock_vector_IS_estimate(theta, data_dict, rl_model):
    """ Calculate the unweighted importance sampling estimate
    on each episodes in the dataframe

    :param theta: The parameter weights
    :type theta: numpy ndarray

    :param dataframe: Contains the episodes
    :type dataframe: pandas dataframe

    :return: A vector of IS estimates calculated for each episode
    :rtype: numpy ndarray(float)
    """
    episodes = data_dict['episodes']
    # weighted_reward_sums_by_episode = data_dict['reward_sums_by_episode']
    ep = episodes[0]

    #start loop in real function
    pi_news = rl_model.get_probs_from_observations_and_actions(theta, ep.states, ep.actions)
    # print("pi news:")
    # print(pi_news)
    pi_ratio_prod = np.prod(pi_news / ep.pis)
    # print("pi_ratio_prod:")
    # print(pi_ratio_prod)
    weighted_return = weighted_sum_gamma(ep.rewards, gamma=0.9)
    # result.append(pi_ratio_prod*weighted_reward_sums_by_episode[ii])
    return pi_ratio_prod * weighted_return
    # end loop in real function


if __name__ == '__main__':
    main()
