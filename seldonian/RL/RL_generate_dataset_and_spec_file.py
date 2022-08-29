from time import time
from seldonian.dataset import RLDataSet
from RLinterface2spec import dataset2spec
from seldonian.RL.hyperparams_and_settings import *
from seldonian.RL.RL_runner import run_trial
from seldonian.utils.RL_utils import *
from seldonian.utils.io_utils import save_pickle


def main():

    hyperparameter_and_setting_dict = define_hyperparameter_and_setting_dict()
    start_time = time()
    episodes, agent = run_trial(hyperparameter_and_setting_dict)
    print(f"data generation took {time() - start_time} seconds")
    # Save episodes to file
    print(len(episodes))
    save_pickle("./episodes_1000episodes.pkl",episodes)
    dataset = RLDataSet(episodes=episodes,meta_information=['O','A','R','pi'])
    # print_return_info(episodes)

    metadata_pth = get_metadata_path(hyperparameter_and_setting_dict["env"])
    save_dir = '.'
    constraint_string = get_constraint_string(hyperparameter_and_setting_dict["env"])
    dataset2spec(save_dir, metadata_pth, dataset, agent.get_policy(), constraint_string)

def get_metadata_path(env_name):
    if env_name == "gridworld":
        return "../../static/datasets/RL/gridworld/gridworld_metadata.json"
    elif env_name == "n_step_mountaincar":
        return "../../static/datasets/RL/mountaincar/n_step_mountaincar_metadata.json"
    elif env_name == "simglucose":
        return "../../static/datasets/RL/simglucose/simglucose.json"
    else:
        error(f"unknown env name {env_name}")

def get_constraint_string(env):
    if env == "gridworld":
        return ['-0.25 - J_pi_new']
    elif env == "n_step_mountaincar":
        return ['-500 - J_pi_new'] #uniform random policy averaged a return of roughly -500 (sample size was 10k episodes)
    elif env == "simglucose":
        return ['-0.25 - J_pi_new'] #needs updating with something reasonable
    else:
        error(f"Unknown env {env}")

def print_return_info(episodes):
    the_sum = 0.0
    the_min = 0.0
    for episode in episodes:
        the_return = sum(episode.rewards)
        if the_return < the_min:
            the_min = the_return
        the_sum += the_return
        print(the_return)
    print(f"\navg return = {the_sum / len(episodes)}")
    print(f"lowest return =  {the_min}")

if __name__ == '__main__':
    main()
