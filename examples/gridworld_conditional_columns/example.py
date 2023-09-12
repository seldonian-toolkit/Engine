# createSpec.py
import autograd.numpy as np
from seldonian.RL.Agents.Policies.Softmax import DiscreteSoftmax
from seldonian.RL.Env_Description.Env_Description import Env_Description
from seldonian.RL.Env_Description.Spaces import Discrete_Space
from seldonian.spec import createRLSpec
from seldonian.dataset import RLDataSet
from seldonian.utils.io_utils import load_pickle

from seldonian.seldonian_algorithm import SeldonianAlgorithm

def main():
	episodes_file = './gridworld_1000episodes.pkl'
	episodes = load_pickle(episodes_file)
	meta_information = {
		'episode_col_names': ['O', 'A', 'R', 'pi_b'],
		'sensitive_col_names': ['M','F']
	}
	M = np.random.randint(0,2,len(episodes))
	F = 1-M
	sensitive_attrs = np.hstack((M.reshape(-1,1),F.reshape(-1,1)))
	dataset = RLDataSet(
		episodes=episodes,
		sensitive_attrs=sensitive_attrs,
		meta_information=meta_information)

	# Initialize policy
	num_states = 9
	observation_space = Discrete_Space(0, num_states-1)
	action_space = Discrete_Space(0, 3)
	env_description =  Env_Description(observation_space, action_space)
	policy = DiscreteSoftmax(hyperparam_and_setting_dict={},
		env_description=env_description)
	env_kwargs={'gamma':0.9}
	save_dir = '.'
	constraint_strs = ['(J_pi_new_IS | [M]) >= -0.25']
	deltas=[0.05]

	spec = createRLSpec(
		dataset=dataset,
		policy=policy,
		constraint_strs=constraint_strs,
		deltas=deltas,
		env_kwargs=env_kwargs,
		save=False,
		save_dir='.',
		verbose=True)

	spec.optimization_hyperparams['num_iters']=20
	spec.optimization_hyperparams['alpha_theta']=0.01
	spec.optimization_hyperparams['alpha_lamb']=0.01
	# Run Seldonian algorithm 
	SA = SeldonianAlgorithm(spec)
	passed_safety,solution = SA.run(debug=True,write_cs_logfile=True)
	if passed_safety:
		print("The solution found is:")
		print(solution)
	else:
		print("No Solution Found")

if __name__ == '__main__':
	np.random.seed(42)
	main()
