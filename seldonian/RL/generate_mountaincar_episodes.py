from time import time
from seldonian.dataset import RLDataSet
from seldonian.RL.hyperparams_and_settings import *
from seldonian.RL.RL_runner import run_trial
from seldonian.utils.RL_utils import *
from seldonian.utils.io_utils import save_pickle
from seldonian.utils.stats_utils import weighted_sum_gamma
from seldonian.spec import createRLSpec

def main():
    """ Run a trial of episodes and save to disk
    """  
    n_episodes = 500
    the_dict = {}
    the_dict["env"] = "n_step_mountaincar"
    the_dict["agent"] = "Parameterized_non_learning_softmax_agent"
    the_dict["basis"] = "Fourier"
    the_dict["order"] = 2
    the_dict["max_coupled_vars"] = -1
    the_dict["num_episodes"] = n_episodes
    the_dict["vis"] = False
    start_time = time()
    episodes, agent = run_trial(the_dict,parallel=True)
    print(len(episodes))
    save_pickle(f"n_step_mountaincar_{n_episodes}episodes.pkl",episodes)
    assert len(episodes) == n_episodes
    print(f"data generation took {time() - start_time} seconds")
    
    # Calculate J, the discounted sum of rewards
    returns = np.array([weighted_sum_gamma(ep.rewards,gamma=1.0) for ep in episodes])
    J = np.mean(returns)
    print(f"J = {J}")


if __name__ == '__main__':
    main()
