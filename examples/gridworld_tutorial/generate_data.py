from functools import partial
import autograd.numpy as np
from seldonian.RL.RL_runner import run_trial
from seldonian.utils.io_utils import save_pickle
from seldonian.RL.environments.gridworld import Gridworld
from seldonian.RL.Agents.Parameterized_non_learning_softmax_agent import Parameterized_non_learning_softmax_agent

def create_env_func():
    return Gridworld(size=3)

def create_agent_func(new_params):   
    dummy_env = Gridworld(size=3)
    env_description = dummy_env.get_env_description()
    agent = Parameterized_non_learning_softmax_agent(
        env_description=env_description,
        hyperparam_and_setting_dict={},
    )
    agent.set_new_params(new_params)
    return agent

def main():
    num_episodes = 1000
    initial_solution = np.zeros((9,4))
    
    hyperparams_and_setting_dict = {}
    hyperparams_and_setting_dict["create_env_func"] = create_env_func
    hyperparams_and_setting_dict["create_agent_func"] = partial(
        create_agent_func,
        new_params=initial_solution
    )
    hyperparams_and_setting_dict["num_episodes"] = num_episodes
    hyperparams_and_setting_dict["num_trials"] = 1
    hyperparams_and_setting_dict["vis"] = False
    episodes = run_trial(hyperparams_and_setting_dict,parallel=True,n_workers=8)

    episodes_file = f'./gridworld_{num_episodes}episodes.pkl'
    save_pickle(episodes_file,episodes)

if __name__ == '__main__':
	main()