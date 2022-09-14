import os, sys
import autograd.numpy as np  # Thinly-wrapped version of Numpy
from time import time

from seldonian.utils.io_utils import load_pickle
from seldonian.dataset import RLDataSet
from seldonian.spec import createRLSpec
from seldonian.RL.Env_Description.Env_Description import (
    Env_Description)
from seldonian.RL.Env_Description.Spaces import (
    Discrete_Space,Continuous_Space)

from seldonian.RL.Agents.Policies.Softmax import Softmax
from seldonian.RL.environments.simglucose_env import Simglucose

if __name__ == '__main__':
    # Load data file
    episode_file = "../../static/datasets/RL/simglucose/simglucose_1000episodes.pkl"

    episodes = load_pickle(episode_file)
    dataset = RLDataSet(episodes=episodes)
    env = Simglucose()
    # Initialize policy
    obs_max = 300.0 
    obs_space_bounds = np.array([[0, obs_max]])
    observation_space = Continuous_Space(obs_space_bounds)
    num_actions = env.num_actions # should be >=5
    assert num_actions == 5
    action_space = Discrete_Space(0, num_actions-1)
    env_description =  Env_Description(observation_space, action_space)
    # Try with same Fourier basis as mountain car
    hyperparam_and_setting_dict = {}
    hyperparam_and_setting_dict["basis"] = "Fourier"
    hyperparam_and_setting_dict["order"] = 2
    hyperparam_and_setting_dict["max_coupled_vars"] = -1

    policy = Softmax(hyperparam_and_setting_dict=hyperparam_and_setting_dict,
        env_description=env_description)
    env_kwargs={'gamma':1.0}
    save_dir = '.'
    constraint_strs = ['J_pi_new >= -6.0']
    deltas=[0.05]

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
        verbose=True
    )
    
