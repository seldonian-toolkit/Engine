from seldonian.RL.Agents.Discrete_Random_Agent import *
from seldonian.RL.environments.gridworld import *

def run_all_trials(hyperparameter_and_setting_dict):
    agent = create_agent(hyperparameter_and_setting_dict)
    num_trials = hyperparameter_and_setting_dict["num_trials"]
    num_episodes = hyperparameter_and_setting_dict["num_episodes"]
    returns = np.zeros((num_trials, num_episodes))
    for trial_num in range(num_trials):
        returns[trial_num] = run_trial(hyperparameter_and_setting_dict, agent)
    return returns

def run_trial(hyperparameter_and_setting_dict, agent):
    env = create_env(hyperparameter_and_setting_dict)
    num_episodes = hyperparameter_and_setting_dict["num_episodes"]
    returns = np.zeros(num_episodes)
    if hyperparameter_and_setting_dict["vis"]:
        env.start_visualizing()
    for episode_num in range(num_episodes):
        returns[episode_num] = run_episode(agent, env)
    return returns

def run_episode(agent, env):
    env.reset()
    observation = env.get_observation()
    episodic_return = 0.0
    while not env.terminated():
        action = agent.choose_action(observation)
        reward = env.transition(action)
        episodic_return += reward
        next_observation = env.get_observation()
        agent.update(observation, next_observation, reward, env.terminated())
    return episodic_return

def create_agent(hyperparameter_and_setting_dict):
    sample_env = create_env(hyperparameter_and_setting_dict)
    env_desc = sample_env.get_env_description()
    agent_type = hyperparameter_and_setting_dict["agent"]
    if agent_type == "discrete_random":
        return Discrete_Random_Agent(env_desc)
    else:
        raise Exception(f"unknown agent type {agent_type}")

def create_env(hyperparameter_and_setting_dict):
    env_type = hyperparameter_and_setting_dict["env"]
    if env_type == "gridworld":
        return Gridworld(3)
    else:
        raise Exception(f"unknown agent type {env_type}")
