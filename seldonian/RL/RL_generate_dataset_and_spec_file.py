from seldonian.RL.environments.gridworld3x3 import Environment
from seldonian.dataset import RLDataSet
from RLinterface2spec import dataset2spec


def main():
    env = Environment()
    episodes = env.generate_data(n_episodes=1000)
    dataset = RLDataSet(episodes=episodes,meta_information=['O','A','R','pi'])
    metadata_pth = "../../static/datasets/RL/gridworld/gridworld3x3_metadata.json"
    save_dir = '.'
    dataset2spec(save_dir, metadata_pth, dataset)

if __name__ == '__main__':
    main()