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
    n_episodes = 1000
    the_dict = {}
    the_dict["env"] = "simglucose"
    the_dict["agent"] = "discrete_random"
    the_dict["num_episodes"] = n_episodes
    the_dict["vis"] = True
    start_time = time()
    episodes, agent = run_trial(the_dict,parallel=True)
    save_pickle(f"simglucose_{n_episodes}episodes.pkl",episodes)
    print(f"data generation took {time() - start_time} seconds")

    max_obs = get_max_obs(episodes)
    print(f"Max observation over {n_episodes} episodes={max_obs}")
    
    # Calculate J, the discounted sum of rewards
    returns = np.array([weighted_sum_gamma(ep.rewards,gamma=1.0) for ep in episodes])
    J = np.mean(returns)
    print(f"J = {J}")


if __name__ == '__main__':
    main()
