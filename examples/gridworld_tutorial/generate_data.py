from seldonian.RL.RL_runner import run_trial
from seldonian.utils.io_utils import save_pickle

hyperparams_and_setting_dict = {}
hyperparams_and_setting_dict["env"] = "gridworld"
hyperparams_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
hyperparams_and_setting_dict["num_episodes"] = 1000
hyperparams_and_setting_dict["num_trials"] = 1
hyperparams_and_setting_dict["vis"] = False

def main():
	episodes, agent = run_trial(hyperparams_and_setting_dict)
	episodes_file = './gridworld_1000episodes.pkl'
	save_pickle(episodes_file,episodes)

if __name__ == '__main__':
	main()