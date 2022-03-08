import numpy as np
import pandas as pd

class DataSetLoader(object):
	def __init__(self,column_names,
		sensitive_column_names,
		regime,**kwargs):
		self.column_names = column_names
		self.sensitive_column_names = sensitive_column_names
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
				regime=self.regime,
				label_column=self.label_column)
		else:
			return DataSet(
				df=df,
				meta_information=self.column_names,
				sensitive_column_names=self.sensitive_column_names,
				regime=regime)

class DataSet(object):
	def __init__(self,df,meta_information,
		regime,label_column='',
		sensitive_column_names=[],**kwargs):
		self.df = df
		self.meta_information = meta_information
		self.regime = regime # supervised or RL
		self.label_column = label_column
		self.sensitive_column_names = sensitive_column_names
	


