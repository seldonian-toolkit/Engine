import numpy as np
import pandas as pd
import ast

measure_functions = [
	'Mean_Error',
	'Mean_Squared_Error',
	'PR',
	'NR',
	'ER',
	'FPR',
	'TPR',
	'FNR',
	'TNR',
]

# map these supported ast operators
# to string representations of operators
op_mapper = {
	ast.Sub: 'sub',
	ast.Add: 'add',
	ast.Mult:'mult',
	ast.Div: 'div',
	ast.Mod: 'modulo',
	ast.Pow: 'pow'
}

# Not supported ast operators
not_supported_op_mapper = {
	ast.BitXor: '^',
	ast.LShift: '<<',
	ast.RShift: '>>',
	ast.BitAnd: '&',
	ast.FloorDiv: '//'
}


class BehavioralConstraint(object):
	def __init__(self,str_expression):
		self.str_expression = str_expression
		
class ScienceGPARegressionConstraint(BehavioralConstraint):
	def __init__(self,str_expression,epsilon):
		super().__init__(str_expression=str_expression)
		self.epsilon = epsilon

	def precalculate_data(self,X,Y,**kwargs):
		dataset = kwargs['dataset']

		male_mask = X.M == 1
		# drop sensitive column names 
		if dataset.sensitive_column_names:
			X = X.drop(columns=dataset.sensitive_column_names)
		X_male = X[male_mask]
		Y_male = Y[male_mask]
		X_female = X[~male_mask]
		Y_female = Y[~male_mask]
		N_male = len(X_male)
		N_female = len(X_female)
		N_least = min(N_male,N_female)
		
		# sample N_least from both without repeats 
		XY_male = pd.concat([X_male,Y_male],axis=1)
		XY_male = XY_male.sample(N_least,replace=True)
		X_male = XY_male.loc[:,XY_male.columns!= dataset.label_column]
		Y_male = XY_male[dataset.label_column]
		
		XY_female = pd.concat([X_female,Y_female],axis=1)
		XY_female = XY_female.sample(N_least,replace=True)
		X_female = XY_female.loc[:,XY_female.columns!= dataset.label_column]
		Y_female = XY_female[dataset.label_column]
		
		data_dict = {
			'X_male':X_male,
			'Y_male':Y_male,
			'X_female':X_female,
			'Y_female':Y_female}
		datasize=N_least
		return data_dict,datasize

	def ghat1(self,model,theta,data_dict):
		# pair up male and female columns and compute a vector of:
		# (y_i - y_hat_i | M) - (y_j - y_hat_j | F) - epsilon
		# There may not be the same number of male and female rows
		# so the number of pairs is min(N_male,N_female)
		X_male = data_dict['X_male']
		Y_male = data_dict['Y_male']
		X_female = data_dict['X_female']
		Y_female = data_dict['Y_female']

		prediction_male = model.predict(theta,X_male)
		mean_error_male = prediction_male-Y_male

		prediction_female = model.predict(theta,X_female)
		mean_error_female = prediction_female-Y_female

		return mean_error_male.values - mean_error_female.values - self.epsilon

	def ghat2(self,model,theta,data_dict):
		# pair up male and female columns and compute a vector of:
		# (y_i - y_hat_i | M) - (y_j - y_hat_j | F) - epsilon
		# There may not be the same number of male and female rows
		# so the number of pairs is min(N_male,N_female)

		X_male = data_dict['X_male']
		Y_male = data_dict['Y_male']
		X_female = data_dict['X_female']
		Y_female = data_dict['Y_female']

		prediction_male = model.predict(theta,X_male)
		mean_error_male = prediction_male-Y_male

		prediction_female = model.predict(theta,X_female)
		mean_error_female = prediction_female-Y_female

		return mean_error_female.values - mean_error_male.values - self.epsilon

custom_ghat_strs = [
	'(y_i - y_hat_i | [M]) - (y_j - y_hat_j | [F]) - epsilon',
	'(y_i - y_hat_i | [F]) - (y_j - y_hat_j | [M]) - epsilon',
]

custom_ghat_dict = {
	'(y_i - y_hat_i | [M]) - (y_j - y_hat_j | [F]) - epsilon': {
		'class':ScienceGPARegressionConstraint,
		'method':'ghat1'},
	'(y_i - y_hat_i | [F]) - (y_j - y_hat_j | [M]) - epsilon': {
	'class':ScienceGPARegressionConstraint,
		'method':'ghat2'},

}
