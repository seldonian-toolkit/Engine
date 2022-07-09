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
			e.g. supervised or RL
		:type regime: str

		"""
		self.regime = regime

	def load_supervised_dataset(self,
		filename,
		metadata_filename,
		include_sensitive_columns=False,
		include_intercept_term=False,
		file_type='csv'):
		""" Create SupervisedDataSet object from file

		:param filename: The file
			containing the data you want to load
		:type filename: str

		:param metadata_filename: The file
			containing the metadata describing the data in filename
		:type metadata_filename: str

		:param file_type: the file extension of filename
		:type file_type: str, defaults to 'csv'
		"""
		if file_type.lower() == 'csv':
			df = pd.read_csv(filename)
		
		elif file_type.lower() == 'pkl' or file_type.lower() == 'pickle':
			df = load_pickle(filename)
		else:
			raise NotImplementedError(f"File type: {file_type} not supported")

		# Load metadata
		metadata_dict = load_json(metadata_filename)

		label_column = metadata_dict['label_column']
		columns = metadata_dict['columns']
		df.columns = columns
		sensitive_column_names = metadata_dict['sensitive_columns']
		return SupervisedDataSet(
			df=df,
			meta_information=columns,
			label_column=label_column,
			sensitive_column_names=sensitive_column_names,
			include_sensitive_columns=include_sensitive_columns,
			include_intercept_term=include_intercept_term)

	def load_RL_dataset(self,
		filename,
		metadata_filename,
		file_type='csv'):
		""" Create RLDataSet object from file

		:param filename: The file
			containing the data you want to load
		:type filename: str

		:param metadata_filename: The file
			containing the metadata describing the data in filename
		:type metadata_filename: str

		:param file_type: the file extension of filename
		:type file_type: str, defaults to 'csv'
		"""

		# Load metadata
		metadata_dict = load_json(metadata_filename)
		columns = metadata_dict['columns']

		if file_type.lower() == 'csv':
			df = pd.read_csv(filename)
		
		elif file_type.lower() == 'pkl' or file_type.lower() == 'pickle':
			df = load_pickle(filename)
		else:
			raise NotImplementedError(f"File type: {file_type} not supported")

		df.columns = columns
		episodes=[]
		
		for episode_index in df.episode_index.unique():
			df_ep = df.loc[df.episode_index==episode_index]
			episode = Episode(states=df_ep.O.values,
							  actions=df_ep.A.values,
							  rewards=df_ep.R.values,
							  pis=df_ep.pi.values)
			episodes.append(episode)
		
		return RLDataSet(
			episodes=episodes,
			meta_information=columns)
		
class DataSet(object):
	def __init__(self,meta_information,
		regime,
		**kwargs):
		""" Object for holding dataframe and dataset metadata

		:param meta_information: list of all column names in the dataframe
		:type meta_information: List(str)

		:param regime: The category of the machine learning algorithm,
			e.g. supervised or RL
		:type regime: str
		"""
		self.meta_information = meta_information
		self.regime = regime 


class SupervisedDataSet(DataSet):
	def __init__(self,df,meta_information,
		label_column,
		sensitive_column_names=[],
		include_sensitive_columns=False,
		include_intercept_term=False,
		**kwargs):
		""" Object for holding Supervised dataframe and dataset metadata
	
		:param df: dataframe containing the full dataset 
		:type df: pandas dataframe

		:param meta_information: list of all column names in the dataframe
		:type meta_information: List(str)

		:param regime: The category of the machine learning algorithm,
			e.g. supervised or RL
		:type regime: str

		:param label_column: The column with the target labels 
			(supervised learning)
		:type label_column: str

		:param sensitive_column_names: The names of the columns that 
			contain the :term:`sensitive attributes<Sensitive attribute>`
		:type sensitive_column_names: List(str)

		:param include_sensitive_columns: Whether to include 
			sensitive columns during training/prediction

		:param include_intercept_term: Whether to add 
			a column of ones as the first column in the dataset.
		"""
		super().__init__(
			meta_information=meta_information,
			regime='supervised')
		self.df = df
		self.label_column = label_column
		self.sensitive_column_names = sensitive_column_names
		self.include_sensitive_columns = include_sensitive_columns
		self.include_intercept_term = include_intercept_term
	
	
class RLDataSet(DataSet):
	def __init__(self,episodes,meta_information,
		**kwargs):
		""" Object for holding RL dataframe and dataset metadata
	
		:param df: dataframe containing the full dataset 
		:type df: pandas dataframe

		:param meta_information: list of all column names in the dataframe
		:type meta_information: List(str)
		"""
		super().__init__(
			meta_information=meta_information,
			regime='RL')
		self.episodes = episodes

class Episode(object):
	def __init__(self,states,actions,rewards,pis):
		""" Object for holding RL episodes
		"""
		self.states = np.array(states)
		self.actions = np.array(actions)
		self.rewards = np.array(rewards)
		self.pis = np.array(pis)

	def __str__(self):
		return f"return = {sum(self.rewards)}\n"+\
	    f"{len(self.states)} states, type of first in array is {type(self.states[0])}: {self.states}\n"\
		+ f"{len(self.actions)} actions, type of first in array is {type(self.actions[0])}: {self.actions}\n"\
		+ f"{len(self.rewards)} rewards, type of first in array is {type(self.rewards[0])}: {self.rewards}\n"\
		+ f"{len(self.pis)} pis, type of first in array is {type(self.pis[0])}: {self.pis}"

