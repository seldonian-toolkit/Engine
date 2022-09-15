from time import time
from seldonian.dataset import RLDataSet
from seldonian.RL.hyperparams_and_settings import *
from seldonian.RL.RL_runner import run_trial
from seldonian.utils.RL_utils import *
from seldonian.utils.io_utils import save_pickle
from seldonian.utils.stats_utils import weighted_sum_gamma
from seldonian.spec import createRLSpec

def get_max_obs(episodes):
    max_obs = 0
    for ii,ep in enumerate(episodes):
        max_obs_this_ep = np.max(ep.observations)
        max_obs = max(max_obs,max_obs_this_ep)
    return max_obs



def main():
    """ Run a trial of episodes and save to disk
    """  
    n_episodes = 100
    the_dict = {}
    the_dict["env"] = "simglucose"
    the_dict["agent"] = "Parameterized_non_learning_softmax_agent"
    the_dict["basis"] = "Fourier"
    the_dict["order"] = 2
    the_dict["max_coupled_vars"] = -1
    the_dict["num_episodes"] = n_episodes
    the_dict["num_trials"] = 1
    the_dict["vis"] = True
    start_time = time()
    solution = np.ones((3,5))
    solution[:,:] = [-1000,0,0,0,0]

    episodes, agent = run_trial(the_dict,parallel=True,model_params=solution)
    
    # save_pickle(f"simglucose_{n_episodes}episodes.pkl",episodes)
    print(f"data generation took {time() - start_time} seconds")

    # max_obs = get_max_obs(episodes)
    # print(f"Max observation over {n_episodes} episodes={max_obs}")
    
    # Calculate J, the discounted sum of rewards
    returns = np.array([weighted_sum_gamma(ep.rewards,gamma=1.0) for ep in episodes])
    J = np.mean(returns)
    print(f"J = {J}")


if __name__ == '__main__':
    main()
