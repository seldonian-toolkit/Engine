# run_experiment.py
from functools import partial
import os
os.environ["OMP_NUM_THREADS"] = "1"
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from concurrent.futures import ProcessPoolExecutor

from seldonian.utils.io_utils import load_pickle
from seldonian.utils.stats_utils import weighted_sum_gamma
from seldonian.RL.RL_runner import run_episode
from seldonian.RL.environments.gridworld import Gridworld
from seldonian.RL.Agents.Parameterized_non_learning_softmax_agent import Parameterized_non_learning_softmax_agent

from experiments.generate_plots import RLPlotGenerator

from createSpec import GridworldSoftmax

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

def generate_episodes_and_calc_J(**kwargs):
    """ Calculate the expected discounted return 
    by generating episodes under the new policy

    :return: episodes, J, where episodes is the list
        of generated ground truth episodes and J is
        the expected discounted return
    :rtype: (List(Episode),float)
    """
    model = kwargs['model']
    num_episodes = kwargs['n_episodes_for_eval']

    # Get trained model weights from running the Seldonian algo
    new_params = model.policy.get_params()

    # Create the env and agent (setting the new policy params) 
    # and run the episodes
    episodes = []
    env = create_env_func()
    agent = create_agent_func(new_params)
    for i in range(num_episodes):
        episodes.append(run_episode(agent,env))

    # Calculate J, the discounted sum of rewards
    returns = np.array([weighted_sum_gamma(ep.rewards,gamma=0.9) for ep in episodes])
    J = np.mean(returns)
    return episodes,J

if __name__ == "__main__":
    # Parameter setup
    np.random.seed(99)
    run_experiments = True
    make_plots = True
    save_plot = False
    include_legend = True

    num_episodes = 1000 # For making trial datasets and for looking up specfile
    n_episodes_for_eval = 1000
    n_trials = 20
    data_fracs = np.logspace(-3,0,10)
    n_workers_for_episode_generation = 1
    n_workers = 8

    frac_data_in_safety = 0.6
    verbose=True
    results_dir = f'results/gridworld_{num_episodes}episodes_{n_trials}trials'
    os.makedirs(results_dir,exist_ok=True)
    plot_savename = os.path.join(results_dir,f'gridworld_experiment.png')
    # Load spec
    specfile = f'./spec.pkl'
    spec = load_pickle(specfile)

    perf_eval_fn = generate_episodes_and_calc_J
    perf_eval_kwargs = {
        'n_episodes_for_eval':n_episodes_for_eval,
        'env_kwargs':spec.model.env_kwargs,
    }
    initial_solution = np.zeros((9,4)) # theta values
    # The setup for generating behavior data for the experiment trials.
    hyperparameter_and_setting_dict = {}
    hyperparameter_and_setting_dict["create_env_func"] = create_env_func
    hyperparameter_and_setting_dict["create_agent_func"] = partial(
        create_agent_func,
        new_params=initial_solution,
    )
    hyperparameter_and_setting_dict["num_episodes"] = num_episodes 
    hyperparameter_and_setting_dict["n_workers_for_episode_generation"] = n_workers_for_episode_generation
    hyperparameter_and_setting_dict["num_trials"] = 1 # Leave as 1 - it is not the same "trial" as experiment trial.
    hyperparameter_and_setting_dict["vis"] = False

    plot_generator = RLPlotGenerator(
        spec=spec,
        n_trials=n_trials,
        data_fracs=data_fracs,
        n_workers=n_workers,
        datagen_method='generate_episodes',
        hyperparameter_and_setting_dict=hyperparameter_and_setting_dict,
        perf_eval_fn=perf_eval_fn,
        perf_eval_kwargs=perf_eval_kwargs,
        results_dir=results_dir,
        )
    
    if run_experiments:
        plot_generator.run_seldonian_experiment(verbose=verbose)

    if make_plots:
        plot_generator.make_plots(fontsize=18,
            performance_label=r"$J(\pi_{\mathrm{new}})$",
            include_legend=include_legend,
            save_format="png",
            title_fontsize=18,
            legend_fontsize=16,
            custom_title=r"$J(\pi_{\mathrm{new}}) \geq -0.25$ (vanilla gridworld 3x3)",
            savename=plot_savename if save_plot else None)


