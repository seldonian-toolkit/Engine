import inspect

from seldonian.RL.Agents.Discrete_Random_Agent import *
from seldonian.RL.Agents.mountain_car_rough_solution import *
from seldonian.RL.Agents.Parameterized_non_learning_softmax_agent import *
from seldonian.RL.Agents.keyboard_gridworld import *

from seldonian.RL.environments.gridworld import *
from seldonian.RL.environments.mountaincar import *
from seldonian.RL.environments.n_step_mountaincar import *

from seldonian.dataset import Episode

import multiprocessing as mp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def run_trial_given_agent_and_env(agent, env, num_episodes):
    """A wrapper for run_trial() where parameters
    are specified explicity rather than via a dictionary.

    Not parallelized here because often agent or env is not pickleable.

    :param agent: RL Agent
    :param env: RL Environment
    :param num_episodes: Number of episodes to run

    :return: List of episodes
    """
    episodes = []

    for episode_num in range(num_episodes):
        episodes.append(run_episode(agent, env))
    return episodes


def run_trial(
    hyperparameter_and_setting_dict, model_params=None, parallel=False, n_workers=8, verbose=False,
):
    """Run a single trial consists of an arbitrary number of episodes.

    :param hyperparameter_and_setting_dict: Specifies the
        environment, agent and number of episodes to run
    :type hyperparameter_and_setting_dict: dict
    :model_params: Policy parameters to set before running the trial
    :parallel: Whether to use parallel processing
    :n_workers: Number of cpus if using parallel processing

    :return: (List of episodes, agent)
    """
    episodes = []
    num_episodes = hyperparameter_and_setting_dict["num_episodes"]
    if verbose:
        print(f"Have {num_episodes} episodes in trial")
    
    if parallel:
        if "create_env_func" in hyperparameter_and_setting_dict:
            create_env_func = hyperparameter_and_setting_dict["create_env_func"]
        else:
            raise RuntimeError(
                "In order to generate episodes in parallel, "
                "you need to specify the 'create_env_func' "
                "in the hyperparameter_and_setting_dict"
            )
        if "create_agent_func" in hyperparameter_and_setting_dict:
            create_agent_func = hyperparameter_and_setting_dict["create_agent_func"]
        else:
            raise RuntimeError(
                "In order to generate episodes in parallel, "
                "you need to specify the 'create_agent_func' "
                "in the hyperparameter_and_setting_dict"
            )

        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp

        chunk_size = num_episodes // n_workers
        chunk_sizes = []
        for i in range(0,num_episodes,chunk_size):
            if (i+chunk_size) > num_episodes:
                chunk_sizes.append(num_episodes-i)
            else:
                chunk_sizes.append(chunk_size)

        create_env_func_list = (create_env_func for _ in range(len(chunk_sizes)))
        create_agent_func_list = (create_agent_func for _ in range(len(chunk_sizes)))

        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp.get_context("fork")
        ) as ex:
            results = tqdm(
                ex.map(
                    run_episodes_par, 
                    create_agent_func_list, 
                    create_env_func_list, 
                    chunk_sizes
                ),
                total=len(chunk_sizes)
            )
            for ep_list in results:
                episodes.extend(ep_list)
        
    else:
        if "create_env_func" in hyperparameter_and_setting_dict:
            create_env_func = hyperparameter_and_setting_dict["create_env_func"]
            env = create_env_func()
        else:
            env = hyperparameter_and_setting_dict["env"]

        if "create_agent_func" in hyperparameter_and_setting_dict:
            create_agent_func = hyperparameter_and_setting_dict["create_agent_func"]
            agent = create_agent_func()
        else:
            if type(hyperparameter_and_setting_dict["agent"]) == str:
                agent = create_agent_fromdict(hyperparameter_and_setting_dict)
            else:
                agent = hyperparameter_and_setting_dict["agent"]

        for _ in range(num_episodes):
            episodes.append(run_episode(agent,env))
    return episodes

def run_episodes_par(
    create_agent_func, create_env_func, num_episodes_this_proc
):
    """Run a bunch of episodes. Function that is run in parallel.

    :param hyperparameter_and_setting_dict: Specifies the
        environment, agent and number of episodes to run
    :type hyperparameter_and_setting_dict: dict
    :model_params: Policy parameters to set before running the trial
    :parallel: Whether to use parallel processing
    :n_workers: Number of cpus if using parallel processing

    :return: (List of episodes, agent)
    """
    np.random.seed()
    env = create_env_func()
    agent = create_agent_func()
    episodes_this_proc = []
    for i in range(num_episodes_this_proc):
        print(f"Episode {i+1}/{num_episodes_this_proc}")
        episodes_this_proc.append(run_episode(agent, env))
    return episodes_this_proc

def run_episode(agent, env):
    """Run a single episode

    :param agent: RL Agent
    :param env: RL Environment

    :return: RL Episode
    :rtype: :py:class:`.Episode`
    """
    observations = []
    actions = []
    rewards = []
    alt_rewards = []
    prob_actions = []

    env.reset()
    observation = env.get_observation()
    while not env.terminated():
        action = agent.choose_action(observation)
        reward = env.transition(action)
        if type(reward) == tuple:
            primary_reward = reward[0]
            alt_rewards_this_step = reward[1:]
            alt_rewards.append(alt_rewards_this_step)
        else:
            primary_reward = reward
        next_observation = env.get_observation()
        agent.update(observation, next_observation, primary_reward, env.terminated())

        observations.append(observation)
        actions.append(action)
        rewards.append(primary_reward)
        prob_actions.append(agent.get_prob_this_action(observation, action))

        observation = next_observation

    return Episode(observations, actions, rewards, prob_actions, np.array(alt_rewards))


def run_episode_from_dict(hyperparameter_and_setting_dict, model_params=None):
    """Run a single episode

    :param agent: RL Agent
    :param env: RL Environment

    :return: RL Episode
    :rtype: :py:class:`.Episode`
    """
    env = hyperparameter_and_setting_dict["env"]
    env_desc = env.get_env_description()
    if type(hyperparameter_and_setting_dict["agent"]) == str:
        agent = create_agent_fromdict(hyperparameter_and_setting_dict)
    else:
        agent = hyperparameter_and_setting_dict["agent"]

    if model_params is not None:
        # print("Setting new params")
        # set agent's weights to the trained model weights
        agent.set_new_params(model_params)
    observations = []
    actions = []
    rewards = []
    prob_actions = []

    env.reset()
    observation = env.get_observation()
    episodic_return = 0.0
    while not env.terminated():
        action = agent.choose_action(observation)
        reward = env.transition(action)
        episodic_return += reward
        next_observation = env.get_observation()
        agent.update(observation, next_observation, reward, env.terminated())

        observations.append(observation)
        actions.append(action)
        rewards.append(reward)
        prob_actions.append(agent.get_prob_this_action(observation, action))

        observation = next_observation

    return Episode(observations, actions, rewards, prob_actions)


def create_agent_fromdict(hyperparameter_and_setting_dict):
    """Create an agent from a dictionary specification

    :param hyperparameter_and_setting_dict: Specifies the
        environment, agent, number of episodes per trial,
        and number of trials

    :return: RL agent
    :rtype: :py:class:`.Agents.Agent`
    """
    env = hyperparameter_and_setting_dict["env"]
    env_desc = env.get_env_description()
    agent_type = hyperparameter_and_setting_dict["agent"]
    if agent_type == "discrete_random":
        return Discrete_Random_Agent(env_desc)
    elif agent_type == "mountain_car_rough_solution":
        return Mountain_car_rough_solution()
    elif agent_type == "Parameterized_non_learning_softmax_agent":
        return Parameterized_non_learning_softmax_agent(
            env_desc, hyperparameter_and_setting_dict
        )
    elif agent_type == "Keyboard_gridworld":
        return Keyboard_gridworld(env_desc)
    else:
        raise Exception(f"unknown agent type {agent_type}")
