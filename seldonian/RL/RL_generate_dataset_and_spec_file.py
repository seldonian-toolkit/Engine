from seldonian.RL.environments.gridworld3x3 import Environment
from time import time
from seldonian.dataset import RLDataSet
from RLinterface2spec import dataset2spec
from seldonian.RL.hyperparams_and_settings import *
from seldonian.RL.RL_runner import run_trial
from utils import *


def main():

    hyperparameter_and_setting_dict = define_hyperparameter_and_setting_dict()
    start_time = time()
    episodes, agent = run_trial(hyperparameter_and_setting_dict)
    print(f"data generation took {time() - start_time} seconds")

    dataset = RLDataSet(episodes=episodes,meta_information=['O','A','R','pi'])

    # print(dataset.episodes[0])
    # print(f"{len(episodes)} episodes")
    metadata_pth = get_metadata_path(hyperparameter_and_setting_dict["env"])
    save_dir = '.'
    constraint_string = get_constraint_string(hyperparameter_and_setting_dict["env"])
    dataset2spec(save_dir, metadata_pth, dataset, agent, constraint_string)

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

if __name__ == '__main__':
    main()
