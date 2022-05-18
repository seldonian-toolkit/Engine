import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pickle
import pandas as pd
from functools import lru_cache, partial
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm

from seldonian.stats_utils import weighted_sum_gamma

class Environment():
	def __init__(self):
		self.gamma = 0.9 # the discount factor when calculating sum of rewards
		self.states = np.arange(9,dtype='int') # 0-8
		self.actions = np.array([0,1,2,3]) # U,D,L,R
		self.reward_dict = {x:0 for x in self.states} # initialize
		self.reward_dict[7]=-1
		self.reward_dict[8]=1

		self.environ_dict = {}
		self.environ_dict[0] = {0:0,1:3,2:0,3:1}
		self.environ_dict[1] = {0:1,1:4,2:0,3:2}
		self.environ_dict[2] = {0:2,1:5,2:1,3:2}
		self.environ_dict[3] = {0:0,1:6,2:3,3:4}
		self.environ_dict[4] = {0:1,1:7,2:3,3:5}
		self.environ_dict[5] = {0:2,1:8,2:4,3:5}
		self.environ_dict[6] = {0:3,1:6,2:6,3:7}
		self.environ_dict[7] = {0:4,1:7,2:6,3:8}
		self.environ_dict[8] = {0:5,1:8,2:7,3:8}
		
		self.initial_state = 0
		self.terminal_state = 8

		self.current_state = self.initial_state

		# initialize parameter weights to all zeros
		# self.initial_weights = np.random.uniform(-1,1,
		#     ((len(self.states)-1)*len(self.actions)
		#     ))
		self.initial_weights = np.zeros(
			(len(self.states)-1)*len(self.actions))
		self.param_weights = self.initial_weights

		# self.param_weights = np.random.uniform(-1,1,
		#     ((len(self.states)-1)*len(self.actions)
		#     ))
		# epsilon=0.5
		# self.param_weights[8] = epsilon

	# @lru_cache
	def denom(self,state):
		return np.sum(np.exp(self.param_weights[state*4+self.actions]))

	# @lru_cache
	def arg(self,state,action):
		return self.param_weights[state*4+action]

	def pi(self,state,action):
		""" Apply the softmax policy to get action probability
		given a state and action 
		param_weights is a flattened parameter vector """
		state = int(state)
		action = int(action)
		
		return np.exp(self.arg(state,action))/self.denom(state)


	def take_step(self):
		# Decide on the action using a policy
		# Need to determine the probability of each action
		# Then draw from that distribution
		
		# episode_entry = np.zeros(4)
		episode_entry = [0,0,0,0]
			
		episode_entry[0] = self.current_state
		
		probs = [self.pi(
			self.current_state,a) for a in self.actions]
		action = np.random.choice(self.actions,p=probs)
		
		episode_entry[1] = action

		# Figure out what the actual probability of selecting that action was
		# action_index = self.actions.index(action)
		action_index = np.where(self.actions==action)[0][0]
		
		prob_thisaction = probs[action_index]
			
		# Figure out next state based on this action
		next_state = self.environ_dict[self.current_state][action]
		# Calculate reward for taking that action
		reward = self.reward_dict[next_state]
		# add to episode entry
		episode_entry[2] = reward
		
		episode_entry[3] = prob_thisaction
			
		# update current state to new state
		self.current_state = next_state
		return episode_entry

   
	def reset(self):
		self.current_state = self.initial_state
		return
		
	def generate_episode(self,return_index=False):
		episode = []
		index = 0
		while self.current_state != self.terminal_state:
			
			entry = self.take_step()
			if return_index:
				entry = np.hstack([index,entry])
				episode.append([entry])
				index+=1
			else:
				episode.append(entry)

		self.reset()
		return episode

	def generate_episodes_par(self,n_episodes=1,return_index=False):
		episodes = []
		for _ in range(n_episodes):
			episode = []
			index = 0
			while self.current_state != self.terminal_state:
				
				entry = self.take_step()
				if return_index:
					entry = np.hstack([index,entry])
					episode.append([entry])
					index+=1
				else:
					episode.append(entry)
			episodes.append(episode)
			self.reset()
		return episodes 

	def generate_data(self,n_episodes,parallel=True,n_workers=8,savename=False,header=False):

		if parallel:
			chunk_size = n_episodes//n_workers
			episodes_per_worker = []
			cumulative_episodes = 0
			for i in range(n_workers):
				if i != n_workers - 1:
					episodes_per_worker.append(chunk_size)
					cumulative_episodes+=chunk_size
				else:
					episodes_per_worker.append(n_episodes-cumulative_episodes)

			helper = partial(self.generate_episodes_par,return_index=False)
			episodes = []
			for n_episodes_this_worker in episodes_per_worker:
			    episodes_this_worker = helper(n_episodes_this_worker)
			    episodes.append(episodes_this_worker)
			# with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('fork')) as executor:
			# 	episodes = tqdm(executor.map(helper, episodes_per_worker),
			# 		total=n_workers)

			data = []
			episode_index = 0
			for list_of_episodes in episodes:
				try:
					for episode in list_of_episodes:
						for entry in episode:
							data.append(np.hstack([episode_index,entry]))
						episode_index+=1
				except Exception as e:
					print(e)
		else:
			episodes = [self.generate_episode(return_index=False) for ii in range(n_episodes)]
			print("done generating episodes")
			data = []
			for episode_i,episode in enumerate(episodes):
				for step in episode:
					data.append([episode_i]+step)


		# df = pd.DataFrame(data,columns=['episode_index','timestep','O','A','R','pi'])
		# print(data)
		df = pd.DataFrame(data,columns=['episode_index','O','A','R','pi'])
		df = df.astype({
			"episode_index": int, 
			# "timestep": int,
			"O": int,
			"A": int,
			"R": float,
			"pi": float
			})

		if savename:
			if savename.endswith('.csv'):
				df.to_csv(savename,index=False,header=header)
			elif savename.endswith('.pkl'):
				with open(savename,'wb') as outfile:
					pickle.dump(df,outfile)
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
