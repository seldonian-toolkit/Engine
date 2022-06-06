import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
import pickle 
from functools import lru_cache, partial
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
from typing import List, Union, Tuple, Dict
from gym import spaces


from seldonian.utils.stats_utils import weighted_sum_gamma


class Basis(object):
	def encode(self, x: np.ndarray)->np.ndarray:
		raise NotImplementedError

	def getNumFeatures(self) -> int:
		raise NotImplementedError

def increment_counter(counter, maxDigit):
	for i in list(range(len(counter)))[::-1]:
		counter[i] += 1
		if (counter[i] > maxDigit):
			counter[i] = 0
		else:
			break

class FourierBasis(Basis):
	def __init__(self, ranges, dorder, iorder, both=False):
		self.iorder = int(iorder)
		self.dorder = int(dorder)
		self.ranges = ranges.astype(np.float64)
		self.feat_range = (ranges[:, 1] - ranges[:, 0]).astype(np.float64)
		self.feat_range[self.feat_range == 0] = 1


		iTerms = iorder * ranges.shape[0]  # number independent terms
		dTerms = pow(dorder+1, ranges.shape[0])  # number of dependent
		oTerms = min(iorder, dorder) * ranges.shape[0]  # number of overlap terms
		self.num_features = int(iTerms + dTerms - oTerms)

		self.both = both
		self.num_input_features = int(ranges.shape[0])
		#print(self.num_input_features, iTerms, dTerms, self.num_features)
		#print("basis ", order, ranges.shape, self.num_features)
		self.C = np.zeros((self.num_features, ranges.shape[0]), dtype=np.float64)
		counter = np.zeros(ranges.shape[0])
		termCount = 0
		while termCount < dTerms:
			for i in range(ranges.shape[0]):
				self.C[termCount, i] = counter[i]
			increment_counter(counter, dorder)
			termCount += 1
		for i in range(ranges.shape[0]):
			for j in range(dorder+1, iorder+1):
				self.C[termCount, i] = j
				termCount += 1

		self.C = self.C.T * np.pi
		if both:
			self.num_features *= 2

	def encode(self, x):
		x = x.flatten().astype(np.float64)
		scaled = (x - self.ranges[:, 0]) / self.feat_range
		dprod = np.dot(scaled, self.C)
		if self.both:
			basis = np.concatenate([np.cos(dprod), np.sin(dprod)])
		else:
			basis = np.cos(dprod)
		return basis

	def getNumFeatures(self):
		if self.both:
			return self.num_features * 2
		else:
			return self.num_features

class Policy(object):
	def get_action(self, obs, stochastic:bool=True) -> Tuple[Union[int, float, np.ndarray, dict], float]:
		raise NotImplementedError

	def grad_logp(self, obs, action) -> Tuple[np.ndarray, float]:
		raise NotImplementedError

	def add_to_params(self, grad: np.ndarray):
		raise NotImplementedError

	def get_params(self)-> np.ndarray:
		'''
		Gets the policy parameters
		:return: numpy vector
		'''
		raise NotImplementedError

	def set_params(self, params: np.ndarray):
		'''
		sets the policy parameters
		:param params: numpy vector for policy
		:return: none
		'''
		raise NotImplementedError

	def get_num_params(self)-> int:
		raise NotImplementedError

class Linear_Softmax(Policy):
	def __init__(self, basis:Basis, n_actions:int):
		super(Linear_Softmax, self).__init__()
		self.basis = basis
		self.n_actions = n_actions
		self.n_inputs =  basis.getNumFeatures()
		self.basis = basis
		self.theta = np.zeros((self.n_inputs, self.n_actions))
		self.num_params = int(self.theta.size)

	def get_action(self, obs:np.ndarray, stochastic:bool=True)->Tuple[int, float]:
		x = self.basis.encode(obs)

		p = self.get_p(x)

		if stochastic:
			a = int(np.random.choice(range(p.shape[0]), p=p, size=1))
			logp = float(np.log(p[a]))
		else:
			a = int(np.argmax(p))
			logp = 0.

		return a, logp

	def log_probabilty(self, obs:np.ndarray, action: int)->float:
		x = self.basis.encode(obs)
		p = self.get_p(x)
		return np.log(p[action])

	def get_num_params(self):
		return self.num_params

	def get_params(self):
		return self.theta.flatten()

	def add_to_params(self, grad):
		self.theta += grad.reshape(self.theta.shape)

	def set_params(self, params):
		self.theta = params.reshape(self.theta.shape)

	def grad_logp(self, obs: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
		x = self.basis.encode(obs)
		theta = self.theta
		a = np.zeros(self.n_actions, dtype=theta.dtype)
		a[action] = 1
		u = self.get_p(x, theta)
		gtheta = x.reshape(-1, 1) * (a - u).reshape(1, -1)  # |s|x1 * 1x|a| -> |s|x|a|
		logp = np.log(u[action])

		return gtheta.flatten(), logp

	def get_p(self, x, theta=None):
		if not isinstance(theta, np.ndarray):
			theta = self.theta
		u = np.exp(np.clip(np.dot(x, theta), -32, 32))
		u /= u.sum()

		return u

class Mountaincar():
	"""
	The cart-pole environment as described in the 687 course material. This
	domain is modeled as a pole balancing on a cart. The agent must learn to
	move the cart forwards and backwards to keep the pole from falling.

	Actions: left (0) and right (1)
	Reward: 1 always

	Environment Dynamics: See the work of Florian 2007
	(Correct equations for the dynamics of the cart-pole system) for the
	observation of the correct dynamics.
	Barrowed from Phil Thomas's RL course Fall 2019. Written by Blossom Metevier and Scott Jordan
	"""

	def __init__(self):
		self._name = "MountainCar"
		self._action = None
		self._reward = 0
		self._isEnd = False
		self.max_timesteps = 1000
		self.min_return = -1*self.max_timesteps
		self.max_return = 0.0

		self._gamma = 1.0
		ranges = np.zeros((2,2))
		ranges[0, :] = [-1.2, 0.5]
		ranges[1, :] = [-0.07, 0.07]
		self.observation_space = spaces.Box(ranges[:, 0], ranges[:, 1], dtype=np.float64)
		self.action_space = spaces.Discrete(3)

		self._x = 0.  # horizontal position of car
		self._v = 0.  # horizontal velocity of the car

		# dynamics
		self._g = 0.0025  # gravity coeff
		self._ucoef = 0.001  # gravity coeff
		self._h = 3.0  # cosine frequency parameter
		self._t = 0.0  # total time elapsed  NOTE: USE must use this variable

	@property
	def isEnd(self) -> bool:
		return self._isEnd

	@property
	def state(self) -> np.ndarray:
		return np.array([self._x, self._v])

	@property
	def gamma(self) -> float:
		return self._gamma

	def nextState(self, state: np.ndarray, action: int) -> np.ndarray:
		"""
		Compute the next state of the pendulum using the euler approximation to the dynamics
		"""
		u = float(action - 2)
		x, v = state
		v += self._ucoef * u - self._g * np.cos(self._h * x)
		v = np.clip(v, self.observation_space.low[1], self.observation_space.high[1])
		x += v
		# If x gets pushed off left boundary then put back on
		# boundary and set velocity to zero
		if x <= self.observation_space.low[0]:
			x = self.observation_space.low[0]
			v = 0.0

		return np.array([x, v])

	def R(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
		"""
		returns a reward for the transition (state,action,next_state)
		"""
		return -1.0

	def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
		"""
		takes one step in the environment and returns the next state, reward, and if it is in the terminal state
		"""
		self._action = action
		state = np.array([self._x, self._v])
		next_state = self.nextState(state, action)
		self._x, self._v = next_state

		self._reward = self.R(state, action, next_state)
		self._t += 1.0
		self._isEnd = self.terminal()

		return self.state, self._reward, self.isEnd

	def reset(self):
		"""
		resets the state of the environment to the initial configuration
		"""
		self._x = -0.5
		self._v = 0.
		self._isEnd = False
		self._t = 0.0
		self._action = None

	def terminal(self) -> bool:
		"""
		terminates the episode if:
			time is greater that 20 seconds
			pole falls |theta| > (pi/12.0)
			cart hits the sides |x| >= 3
		"""
		if self._t >= self.max_timesteps:
			return True
		if self._x >= 0.05:
			return True
		return False


class Environment(Mountaincar):
	def __init__(self):
		super().__init__()
		# gamma is set to 1 in Mountaincar
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
		self.basis = FourierBasis(ranges, dorder=dorder, iorder=iorder, both=both)
		self.num_actions = self.action_space.n  # assumes actions space is discrete. Continuous actions can also be handled by policy-critic

		self.policy = Linear_Softmax(self.basis, self.num_actions)
		self.num_params = self.policy.get_num_params()
		self.initial_weights = np.ones(self.num_params)
		# self.initial_weights = np.zeros(self.num_params)

		
	@property
	def param_weights(self): 
		return self._param_weights

	@param_weights.setter
	def param_weights(self, value): 
		""" Sets the value of param_weights and also updates the policy 
		parameter weights, theta """
		self._param_weights = value
		self.policy.theta = value.reshape(self.policy.n_inputs, self.policy.n_actions)


	def generate_episode(self):
		
		episode = []
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
			row = [s,a,r,np.exp(logp)]
			episode.append(row)
			s = snext  # updates the current state to be the next state
	
		return episode

	def generate_episodes_par(self,n_episodes=1):
		episodes = []
		for episode_i in range(n_episodes):
			episode = []
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
				row = [s,a,r,np.exp(logp)]
				episode.append(row)
				s = snext  # updates the current state to be the next state
			episodes.append(episode)
		return episodes 

	def generate_data(self,n_episodes,parallel=True,n_workers=8,savename=False,header=False):
		"""
		"""

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
			
			with ProcessPoolExecutor(max_workers=n_workers,
				mp_context=mp.get_context('fork')) as executor:
				episodes = tqdm(executor.map(
					self.generate_episodes_par, episodes_per_worker),
					total=n_workers)

			episode_index = 0
			data = []
			for list_of_episodes in episodes:
				try:
					for episode in list_of_episodes:
						for entry in episode:
							data.append(np.hstack([episode_index,entry]))
						episode_index+=1
				except Exception as e:
					print(e)
		else:
			episodes = [self.generate_episode() for ii in range(n_episodes)]
			print("done generating episodes")
			data = []
			for episode_i,episode in enumerate(episodes):
				for entry in episode:
					data.append([episode_i]+entry)
					
		df = pd.DataFrame(data,columns=['episode_index','O','A','R','pi'])
		df = df.astype({
			"episode_index": int, 
			"A": int,
			"R": float,
			"pi": float
			})

		if savename:
			with open(savename,'wb') as outfile:
				pickle.dump(df,outfile)
			# df.to_csv(savename,index=False,header=header)
				print(f"Saved {savename}")
		return df 


	def calc_J_from_df(self,df):
		""" Given a dataset and gamma 
		calculate the expected return of the sum 
		of discounted rewards."""
		ws_helper = partial(weighted_sum_gamma,gamma=self.gamma)

		discounted_sum_rewards_episodes=df.groupby(
			'episode_index')['R'].apply(ws_helper)
		# Normalize to [0,1]
		normalized_returns = (discounted_sum_rewards_episodes\
			-self.min_return)/(self.max_return-self.min_return)
		return np.mean(normalized_returns)
