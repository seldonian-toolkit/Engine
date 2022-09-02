from time import time
from seldonian.dataset import RLDataSet
from seldonian.RL.hyperparams_and_settings import *
from seldonian.RL.RL_runner import run_trial
from seldonian.utils.RL_utils import *
from seldonian.utils.io_utils import save_pickle
from seldonian.spec import createRLSpec


def main():

    hyperparameter_and_setting_dict = define_hyperparameter_and_setting_dict()
    start_time = time()
    episodes, agent = run_trial(hyperparameter_and_setting_dict)
    # Save episodes as pkl file:
    save_pickle("n_step_mountaincar_100episodes.pkl",episodes)
    print(f"data generation took {time() - start_time} seconds")
    dataset = RLDataSet(episodes=episodes,meta_information=['O','A','R','pi'])

    env_name = hyperparameter_and_setting_dict["env"]
    metadata_pth = get_metadata_path(env_name)
    save_dir = '.'
    constraint_strs = get_constraint_string(env_name)
    deltas = [0.05]
    env_kwargs = get_env_kwargs(env_name)
    policy = agent.get_policy()
    createRLSpec(
        dataset=dataset,
        policy=policy,
        constraint_strs=constraint_strs,
        deltas=deltas,
        env_kwargs=env_kwargs,
        frac_data_in_safety=0.6,
        initial_solution_fn=None,
        use_builtin_primary_gradient_fn=False,
        save=True,
        save_dir=save_dir,
        verbose=False)

def get_metadata_path(env_name):
    if env_name == "gridworld":
        return "../../static/datasets/RL/gridworld/gridworld_metadata.json"
    elif env_name == "n_step_mountaincar":
        return "../../static/datasets/RL/n_step_mountaincar/n_step_mountaincar_metadata.json"
    elif env_name == "simglucose":
        return "../../static/datasets/RL/simglucose/simglucose.json"
    else:
        error(f"unknown env name {env_name}")

def get_constraint_string(env_name):
    if env_name == "gridworld":
        return ['J_pi_new >= -0.25']
    elif env_name == "n_step_mountaincar":
        return ['J_pi_new >= -500'] #uniform random policy averaged a return of roughly -500 (sample size was 10k episodes)
    elif env_name == "simglucose":
        return ['zzzz'] #needs updating with something reasonable
    else:
        error(f"Unknown env_name {env_name}")

def get_env_kwargs(env_name):
    if env_name == "gridworld":
        from seldonian.RL.environments.gridworld import Gridworld
        RL_environment = Gridworld()
        gamma = RL_environment.gamma
        return {'gamma':gamma}
    elif env_name == "n_step_mountaincar":
        from seldonian.RL.environments.n_step_mountaincar import N_step_mountaincar
        RL_environment = N_step_mountaincar()
        gamma = RL_environment.gamma
        return {'gamma':gamma}
    elif env_name == "simglucose":
        return {} #needs updating 
    else:
        error(f"Unknown env_name {env_name}")

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
