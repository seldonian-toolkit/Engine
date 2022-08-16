from seldonian.RL.environments.gridworld3x3 import Environment
from time import time
from seldonian.dataset import RLDataSet
from RLinterface2spec import dataset2spec
from seldonian.RL.RL_runner import run_trial
from utils import *

hyperparams_and_setting_dict = {}
hyperparams_and_setting_dict["env"] = "gridworld"
hyperparams_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
hyperparams_and_setting_dict["num_episodes"] = 1000
hyperparams_and_setting_dict["num_trials"] = 1

def main():
    metadata_pth = "../../static/datasets/RL/gridworld/gridworld_metadata.json"
    episodes, agent = run_trial(hyperparameter_and_setting_dict)
    print(f"data generation took {time() - start_time} seconds")

    dataset = RLDataSet(episodes=episodes,meta_information=['O','A','R','pi'])

    metadata_pth = get_metadata_path(hyperparameter_and_setting_dict["env"])
    save_dir = '.'
    constraint_strs = ['-0.25 - J_pi_new']
    dataset2spec(save_dir, metadata_pth, dataset, agent, constraint_string)


if __name__ == '__main__':
    main()
