from seldonian.RL.environments.gridworld3x3 import Environment
from seldonian.dataset import RLDataSet
from RLinterface2spec import dataset2spec
from seldonian.RL.hyperparams_and_settings import *
from seldonian.RL.RL_runner import run_trial


def main():
    # env = Environment()
    # episodes = env.generate_data(n_episodes=1000)

    hyperparameter_and_setting_dict = define_hyperparameter_and_setting_dict()
    episodes, agent = run_trial(hyperparameter_and_setting_dict)

    dataset = RLDataSet(episodes=episodes,meta_information=['O','A','R','pi'])
    print(dataset.episodes[0])
    print(f"{len(episodes)} episodes")
    metadata_pth = "../../static/datasets/RL/gridworld/gridworld_metadata.json"
    save_dir = '.'
    dataset2spec(save_dir, metadata_pth, dataset, agent)

if __name__ == '__main__':
    main()
