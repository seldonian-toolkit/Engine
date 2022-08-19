from seldonian.RL.RL_runner import run_trial
from seldonian.spec import createRLSpec
from seldonian.dataset import RLDataSet

hyperparams_and_setting_dict = {}
hyperparams_and_setting_dict["env"] = "gridworld"
hyperparams_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
hyperparams_and_setting_dict["num_episodes"] = 1000
hyperparams_and_setting_dict["num_trials"] = 1
hyperparams_and_setting_dict["vis"] = False

def main():
	episodes, agent = run_trial(hyperparams_and_setting_dict)
	dataset = RLDataSet(episodes=episodes)

	metadata_pth = "../../static/datasets/RL/gridworld/gridworld_metadata.json"
	save_dir = '.'
	constraint_strs = ['J_pi_new >= -0.25']
	deltas=[0.05]

	createRLSpec(
		dataset=dataset,
		metadata_pth=metadata_pth,
		agent=agent,
		constraint_strs=constraint_strs,
		deltas=deltas,
		save_dir=save_dir,
		verbose=True)

if __name__ == '__main__':
	main()
