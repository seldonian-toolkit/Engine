import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd


class DataSetLoader(object):
	def __init__(self,
		regime,
		column_names,
		sensitive_column_names,
		include_sensitive_columns=False,
		include_intercept_term=False,
		scale_features=False,
		**kwargs):
		self.column_names = column_names
		self.sensitive_column_names = sensitive_column_names
		self.include_sensitive_columns = include_sensitive_columns
		self.include_intercept_term = include_intercept_term
		self.scale_features = scale_features
		self.regime = regime
		if self.regime == 'supervised':
			self.label_column = kwargs['label_column']

	def from_csv(self,csv_file):
		df = pd.read_csv(csv_file,names=self.column_names)
		if self.regime == 'supervised':
			return DataSet(
				df=df,
				meta_information=self.column_names,
				sensitive_column_names=self.sensitive_column_names,
				include_sensitive_columns=self.include_sensitive_columns,
				include_intercept_term=self.include_intercept_term,
				scale_features=self.scale_features,
				regime=self.regime,
				label_column=self.label_column)
		elif self.regime == 'RL':
			return DataSet(
				df=df,
				meta_information=self.column_names,
				regime=self.regime)

class DataSet(object):
	def __init__(self,df,meta_information,
		regime,label_column='',
		sensitive_column_names=[],
		include_sensitive_columns=False,
		include_intercept_term=False,
		scale_features=False,
		**kwargs):
		self.df = df
		self.meta_information = meta_information
		self.regime = regime # supervised or RL
		self.label_column = label_column
		self.sensitive_column_names = sensitive_column_names
		self.include_sensitive_columns = include_sensitive_columns
		self.include_intercept_term = include_intercept_term
		self.scale_features=scale_features
	
	
	


