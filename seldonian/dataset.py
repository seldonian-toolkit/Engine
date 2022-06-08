""" Build and load Datasets for running Seldonian algorithms """

import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd


class DataSetLoader(object):
	def __init__(self,
		regime,
		column_names,
		sensitive_column_names,
		include_sensitive_columns=False,
		include_intercept_term=False,
		**kwargs):
		""" Object for loading datasets from disk into DataSet objects
		
		:param regime: The category of the machine learning algorithm,
			e.g. supervised or RL
		:type regime: str

		:param column_names: list of all column names in the dataset.
		:type column_names: List(str)

		:param sensitive_column_names: The names of the columns that 
			contain the :term:`sensitive attributes<Sensitive attribute>`.
		:type sensitive_column_names: List(str)

		:param include_sensitive_columns: Whether to include 
			sensitive columns during training/prediction
			(supervised learning).
		:param include_intercept_term: Whether to add 
			a column of ones as the first column in the dataset
			(supervised learning).

		:ivar label_column: The column with the target labels 
			(supervised learning)
		:vartype label_column: str
		"""
		self.column_names = column_names
		self.sensitive_column_names = sensitive_column_names
		self.include_sensitive_columns = include_sensitive_columns
		self.include_intercept_term = include_intercept_term
		self.regime = regime
		if self.regime == 'supervised':
			self.label_column = kwargs['label_column']

	def from_csv(self,csv_file):
		""" Create DataSet object from csv file

		:param csv_file: The filename of the csv file 
			containing the data you want to load
		:type csv_file: str
		"""
		df = pd.read_csv(csv_file,names=self.column_names)
		if self.regime == 'supervised':
			return SupervisedDataSet(
				df=df,
				meta_information=self.column_names,
				sensitive_column_names=self.sensitive_column_names,
				include_sensitive_columns=self.include_sensitive_columns,
				include_intercept_term=self.include_intercept_term,
				regime=self.regime,
				label_column=self.label_column)
		elif self.regime == 'RL':
			return RLDataSet(
				df=df,
				meta_information=self.column_names,
				regime=self.regime)

	def from_pickle(self,pkl_file):
		""" Create DataSet object from pickle file

		:param pkl_file: The filename of the pickle file 
			containing the data you want to load
		:type pkl_file: str
		"""
		import pickle
		with open(pkl_file,'rb') as infile:
			df = pickle.load(infile) # will include the header if it is present
		if self.regime == 'supervised':
			return DataSet(
				df=df,
				meta_information=self.column_names,
				sensitive_column_names=self.sensitive_column_names,
				include_sensitive_columns=self.include_sensitive_columns,
				include_intercept_term=self.include_intercept_term,
				regime=self.regime,
				label_column=self.label_column)
		elif self.regime == 'RL':
			return DataSet(
				df=df,
				meta_information=self.column_names,
				regime=self.regime)

class DataSet(object):
	def __init__(self,df,meta_information,
		regime,
		**kwargs):
		""" Object for holding dataframe and dataset metadata
	
		:param df: dataframe containing the full dataset 
		:type df: pandas dataframe

		:param meta_information: list of all column names in the dataframe
		:type meta_information: List(str)

		:param regime: The category of the machine learning algorithm,
			e.g. supervised or RL
		:type regime: str
		"""
		self.df = df
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
		super().__init__(df=df,
			meta_information=meta_information,
			regime='supervised')

		self.label_column = label_column
		self.sensitive_column_names = sensitive_column_names
		self.include_sensitive_columns = include_sensitive_columns
		self.include_intercept_term = include_intercept_term
	
	
class RLDataSet(DataSet):
	def __init__(self,df,meta_information,
		**kwargs):
		""" Object for holding RL dataframe and dataset metadata
	
		:param df: dataframe containing the full dataset 
		:type df: pandas dataframe

		:param meta_information: list of all column names in the dataframe
		:type meta_information: List(str)
		"""
		super().__init__(df=df,
			meta_information=meta_information,
			regime='RL')

