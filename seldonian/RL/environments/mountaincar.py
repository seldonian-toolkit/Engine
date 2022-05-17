import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
from functools import lru_cache, partial
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm

from seldonian.stats_utils import weighted_sum_gamma

from rllib.environments import Mountaincar
from rllib.policies import Linear_Softmax
from rllib.basis import FourierBasis


class Environment(Mountaincar):
	def __init__(self):
		super().__init__()
		print(f"gamma={self.gamma}")
		self.action_repeat = 20
		ranges = np.array([self.observation_space.low, self.observation_space.high]).T
		# ranges should be a (n,2) numpy array where each row contains the min and max values for that state variable
		# the ranges are necessary for the fourier basis because it works on inputs in the range [0,1].

		dorder = 2  # this is the max order to combine different state variables in the fourier basis,
					# e.g., one feature would be cos(3x[0] + 4x[1]) if there were two state variables and dorder >= 4
					# all coefficients are used. The number of basis functions for this is pow(dorder+1, ranges.shape[0]).
		iorder = 2  # this is the max order for each each state variable applied independently of other variables.
					# e.g., order three would have the features cos(1x[0]), cos(2x[0]), cos(3x[0]) for each state variable in x
					# The number of basis functions for this component is ioder*ranges.shape[0].
					# This term is ignored if dorder >= iorder.
		both = False  # If true then both sine and cosine are used to create features
		basis = FourierBasis(ranges, dorder=dorder, iorder=iorder, both=both)
		num_actions = self.action_space.n  # assumes actions space is discrete. Continuous actions can also be handled by policy-critic

		self.policy = Linear_Softmax(basis, num_actions)
		self.num_params = self.policy.get_num_params()
		self.initial_weights = np.zeros(self.num_params)
		self.param_weights = self.initial_weights

	def generate_data(self,n_episodes,parallel=True,n_workers=8,savename=False,header=False):
		"""
		"""

		data = []
		for episode_i in range(n_episodes):
			# traj = memory.new()
			self.reset()  # reset environment for initial episode
			s = self.state  # gets the first state features
			done = False  # flag to tell when an episode ends
			while not done:  # repeat until the end of the episode
				a, logp = self.policy.get_action(s)  # gets the action and the log probability of that action for the current state
				r = 0.0
				for i in range(self.action_repeat):
					snext, reward, done = self.step(a)  # gets the next state, reward, and whether or note the episode is over.
					r += reward
					if done:  # if episode is over break the action repeat
						break
				row = [episode_i,s,a,r,np.exp(logp)]
				data.append(row)
				# memory.add(traj, s, a, r, logp, snext, done)  # saves the transition into a buffer (not an experience replay style buffer)
				s = snext  # updates the current state to be the next state

			# returns.append(G)  # add return from this episode to the list
			
		df = pd.DataFrame(data,columns=['episode_index','O','A','R','pi'])
		df = df.astype({
			"episode_index": int, 
			"A": int,
			"R": float,
			"pi": float
			})

		if savename:
			df.to_csv(savename,index=False,header=header)
			print(f"Saved {savename}")
		return df 


	def calc_J_from_df(self,df,gamma=0.9):
		""" Given a dataset and gamma 
		calculate the expected return of the sum 
		of discounted rewards."""
		ws_helper = partial(weighted_sum_gamma,gamma=gamma)
		discounted_sum_rewards_episodes=df.groupby(
			'episode_index')['R'].apply(ws_helper)
		return np.mean(discounted_sum_rewards_episodes)
