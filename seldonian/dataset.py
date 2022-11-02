""" Build and load datasets for running Seldonian algorithms """

import autograd.numpy as np
import pandas as pd
import pickle
from seldonian.utils.io_utils import load_json,load_pickle

class DataSetLoader():
	def __init__(self,
		regime,
		**kwargs):
		""" Object for loading datasets from disk into DataSet objects
		
		:param regime: The category of the machine learning algorithm,
			e.g., supervised_learning or reinforcement_learning
		:type regime: str
		"""
		self.regime = regime

	def load_supervised_dataset(self,
		filename,
		metadata_filename,
		file_type='csv'):
		""" Create SupervisedDataSet object from file

		:param filename: The file
			containing the features, labels and sensitive attributes
		:type filename: str
		:param metadata_filename: The file
			containing the metadata describing the data in filename
		:type metadata_filename: str
		:param file_type: the file extension of filename
		:type file_type: str, defaults to 'csv'
		"""
		# Load metadata
		(regime, sub_regime, all_col_names,
			feature_col_names,label_col_names,
			sensitive_col_names) = load_supervised_metadata(metadata_filename)
		
		# infer feature column names 
		meta_information = {}
		meta_information['feature_col_names'] = feature_col_names
		meta_information['label_col_names'] = label_col_names
		meta_information['sensitive_col_names'] = sensitive_col_names
		meta_information['sub_regime'] = sub_regime

		if file_type.lower() == 'csv':
			df = pd.read_csv(filename,header=None,names=all_col_names)
			# separate out features, labels, and sensitive attrs
			features = df.loc[:,feature_col_names].values
			labels = np.squeeze(df.loc[:,label_col_names].values) # converts shape from (N,1) -> (N,) if only a single label column.
			sensitive_attrs = df.loc[:,sensitive_col_names].values
			num_datapoints = len(df)
		else:
			raise NotImplementedError(f"File type: {file_type} not supported")
	
		return SupervisedDataSet(
			features=features,
			labels=labels,
			sensitive_attrs=sensitive_attrs,
			num_datapoints=num_datapoints,
			meta_information=meta_information)

	def load_RL_dataset_from_csv(self,
		filename,metadata_filename=None):
		""" Create RLDataSet object from file
		containing the episodes as a CSV with format:
		episode_index,obs,action,reward,probability_of_action.

		:param filename: The file
			containing the data you want to load
		:type filename: str
		:param metadata_filename: Name of metadata file
		:type metadata_filename: str
		"""

		# Load metadata
		if metadata_filename:
			metadata_dict = load_json(metadata_filename)
			column_names = metadata_dict['columns']
		else:
			column_names = ['episode_index','O','A','R','pi_b']

		df = pd.read_csv(filename,header=None)
		df.columns = column_names
		episodes=[]
		
		for episode_index in df.episode_index.unique():
			df_ep = df.loc[df.episode_index==episode_index]
			episode = Episode(observations=df_ep.iloc[:,1].values,
							  actions=df_ep.iloc[:,2].values,
							  rewards=df_ep.iloc[:,3].values,
							  action_probs=df_ep.iloc[:,4].values)
			episodes.append(episode)
		
		return RLDataSet(
			episodes=episodes,
			meta_information=column_names)
	
	def load_RL_dataset_from_episode_file(self,
		filename):
		""" Create RLDataSet object from file 

		:param filename: The file
			containing the pickled lists of :py:class:`.Episode` objects
		:type filename: str
		"""

		columns = ["O","A","R","pi_b"]

		episodes = load_pickle(filename)
		
		return RLDataSet(
			episodes=episodes,
			meta_information=columns)
		
class DataSet(object):
	def __init__(self,
		meta_information,
		regime,
		**kwargs):
		""" Object for holding dataframe and dataset metadata

		:param meta_information: list of all column names in the dataframe
		:type meta_information: List(str)

		:param regime: The category of the machine learning algorithm,
			e.g., supervised_learning or reinforcement_learning
		:type regime: str
		"""
		self.meta_information = meta_information
		self.regime = regime 

class SupervisedDataSet(DataSet):
	def __init__(self,
		features,
		labels,
		sensitive_attrs,
		num_datapoints,
		meta_information):
		super().__init__(
			meta_information=meta_information,
			regime='supervised_learning')

		self.features = features
		self.labels = labels
		self.sensitive_attrs = sensitive_attrs
		self.num_datapoints = num_datapoints
		
		self.feature_col_names = meta_information['feature_col_names']
		self.label_col_names = meta_information['label_col_names']
		self.sensitive_col_names = meta_information['sensitive_col_names']
		
		self.n_features = len(self.feature_col_names)
		self.n_labels = len(self.label_col_names)
		self.n_sensitive_attrs = len(self.sensitive_col_names)
	
class RLDataSet(DataSet):
	def __init__(self,episodes,meta_information=['O','A','R','pi_b'],
		**kwargs):
		""" Object for holding RL dataframe and dataset metadata
	
		:param episodes: List of episodes
		:type episodes: list(:py:class:`.Episode`)
		:param meta_information: List of attribute names in each Episode,
			e.g. ['o','a','r','pi_b']
		:type meta_information: list(str)
		"""
		super().__init__(
			meta_information=meta_information,
			regime='reinforcement_learning')
		self.episodes = episodes


class Episode(object):
	def __init__(self,observations,actions,rewards,action_probs):
		""" Object for holding RL episodes
		
		:param observations: List of observations for each timestep
			in the episode
		:param actions: List of actions 
		:param rewards: List of rewards 
		:param action_probs: List of action probabilities 
			from the behavior policy
		"""
		self.observations = np.array(observations)
		self.actions = np.array(actions)
		self.rewards = np.array(rewards)
		self.action_probs = np.array(action_probs)

	def __str__(self):
		return f"return = {sum(self.rewards)}\n"+\
		f"{len(self.observations)} observations, type of first in array is {type(self.observations[0])}: {self.observations}\n"\
		+ f"{len(self.actions)} actions, type of first in array is {type(self.actions[0])}: {self.actions}\n"\
		+ f"{len(self.rewards)} rewards, type of first in array is {type(self.rewards[0])}: {self.rewards}\n"\
		+ f"{len(self.action_probs)} action_probs, type of first in array is {type(self.action_probs[0])}: {self.prob_actions}"

def load_supervised_metadata(filename):
    """ Load metadata from JSON file into a dictionary

    :param filename: The file to load
    """
    metadata_dict = load_json(filename)
    regime = metadata_dict['regime']
    assert regime == 'supervised_learning'
    sub_regime = metadata_dict['sub_regime']
    assert sub_regime in ['regression','classification',
    	'binary_classification','multiclass_classification']
    all_col_names = metadata_dict['all_col_names']
    label_col_names = metadata_dict['label_col_names']
    sensitive_col_names = metadata_dict['sensitive_col_names']
    # infer feature column names - keep order same
    feature_col_names = [x for x in all_col_names if (x not in label_col_names) and (x not in sensitive_col_names)]
    return regime, sub_regime, all_col_names, feature_col_names, label_col_names, sensitive_col_names