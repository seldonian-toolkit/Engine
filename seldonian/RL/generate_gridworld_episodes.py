from time import time
from seldonian.dataset import RLDataSet
from seldonian.RL.hyperparams_and_settings import *
from seldonian.RL.RL_runner import run_trial
from seldonian.utils.RL_utils import *
from seldonian.utils.io_utils import save_pickle
from seldonian.utils.stats_utils import weighted_sum_gamma
from seldonian.spec import createRLSpec
from seldonian.RL.environments.gridworld import Gridworld

def main():
    """ Run a trial of episodes and save to disk
    """  
    n_episodes = 1000
    the_dict = {}
    the_dict["env"] = Gridworld()
    the_dict["agent"] = "Parameterized_non_learning_softmax_agent"
    the_dict["num_episodes"] = n_episodes
    the_dict["vis"] = False
    start_time = time()
    episodes, agent = run_trial(the_dict,parallel=False)
    print(len(episodes))
    save_pickle(f"gridworld_{n_episodes}episodes.pkl",episodes,verbose=True)
    assert len(episodes) == n_episodes
    print(f"data generation took {time() - start_time} seconds")
    
    # # Calculate J, the discounted sum of rewards
    returns = np.array([weighted_sum_gamma(ep.rewards,gamma=0.9) for ep in episodes])
    J = np.mean(returns)
    print(f"J = {J}")


if __name__ == '__main__':
    main()
