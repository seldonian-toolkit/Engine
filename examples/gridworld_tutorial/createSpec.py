# createSpec.py
import autograd.numpy as np
from seldonian.RL.Agents.Policies.Softmax import DiscreteSoftmax
from seldonian.RL.Env_Description.Env_Description import Env_Description
from seldonian.RL.Env_Description.Spaces import Discrete_Space
from seldonian.spec import createRLSpec
from seldonian.dataset import RLDataSet,RLMetaData
from seldonian.utils.io_utils import load_pickle

class GridworldSoftmax(DiscreteSoftmax):
    def __init__(self, env_description):
        hyperparam_and_setting_dict = {}
        super().__init__(hyperparam_and_setting_dict, env_description)

    def get_probs_from_observations_and_actions(self,observations,actions,_):
        return self.softmax(self.FA.weights)[observations,actions]

    def softmax(self,x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

def main():
    episodes_file = './gridworld_1000episodes.pkl'
    episodes = load_pickle(episodes_file)
    meta = RLMetaData(all_col_names=["episode_index", "O", "A", "R", "pi_b"])
    dataset = RLDataSet(episodes=episodes,meta=meta)

    # Initialize policy
    num_states = 9
    observation_space = Discrete_Space(0, num_states-1)
    action_space = Discrete_Space(0, 3)
    env_description =  Env_Description(observation_space, action_space)
    policy = GridworldSoftmax(env_description=env_description)
    env_kwargs={'gamma':0.9}
    save_dir = '.'
    constraint_strs = ['J_pi_new_IS >= -0.25']
    deltas=[0.05]

    spec = createRLSpec(
        dataset=dataset,
        policy=policy,
        constraint_strs=constraint_strs,
        deltas=deltas,
        env_kwargs=env_kwargs,
        save=True,
        save_dir='.',
        verbose=True)

if __name__ == '__main__':
    main()