import numpy as np
from sklearn.linear_model import (LinearRegression,
	LogisticRegression, SGDClassifier)
from seldonian.stats_utils import weighted_sum_gamma
from functools import partial

class SeldonianModel(object):
	def __init__(self):
		pass


class SupervisedModel(SeldonianModel):
	def __init__(self):
		pass

	def fit(self,X,Y):
		reg = self.model_class().fit(X, Y)
		return np.hstack([np.array(reg.intercept_),reg.coef_[1:]])


class RegressionModel(SupervisedModel):
	def __init__(self):
		super().__init__()

	def sample_from_statistic(self,
		statistic_name,model,theta,data_dict):
		
		if statistic_name == 'Mean_Squared_Error':
			return model.vector_Mean_Squared_Error(model,
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'Mean_Error':
			return model.vector_Mean_Error(model,
				theta,data_dict['features'],data_dict['labels'])

		raise NotImplementedError(f"Statistic: {statistic_name} is not implemented")

	def sample_Mean_Squared_Error(self,model,theta,X,Y):
		"""
		Calculate sample mean squared error given a solution and 
		data. Put under regressionmodel
		"""
		n = len(X)
		prediction = model.predict(theta,X) # vector of values
		res = sum(pow(prediction-Y,2))/n
		return res

	def sample_Mean_Error(self,model,theta,X,Y):
		"""
		Calculate sample mean error given a solution and 
		data. Put under regressionmodel
		"""
		n = len(X)
		prediction = model.predict(theta,X) # vector of values
		res = sum(prediction-Y)/n
		return res

	def vector_Mean_Squared_Error(self,model,theta,X,Y):
		""" Calculate sample mean squared error at each 
		point in the dataset and return as a 1-vector
		"""  
		prediction = model.predict(theta, X)
		return pow(prediction-Y,2)
		
	def vector_Mean_Error(self,model,theta,X,Y):
		""" Calculate sample mean error at each 
		point in the dataset and return as a 1-vector
		"""  
		prediction = model.predict(theta, X)
		return prediction-Y


class LinearRegressionModel(RegressionModel):
	def __init__(self):
		super().__init__()
		self.model_class = LinearRegression

	def predict(self,theta,X):
		""" Given a set of weights, theta,
		and an array of feature vectors, X, which include offsets
		in the first column,
		make prediction using the model """
		return np.dot(theta.T,X.T)


class ClassificationModel(SupervisedModel):
	def __init__(self):
		super().__init__()

	def sample_from_statistic(self,
		statistic_name,model,theta,data_dict):
		if statistic_name == 'PR':
			return model.Vector_Positive_Rate(model,
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'NR':
			return model.Vector_Negative_Rate(model,
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'FPR':
			return model.Vector_False_Positive_Rate(model,
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'FNR':
			return model.Vector_False_Negative_Rate(model,
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'TPR':
			return model.Vector_True_Positive_Rate(model,
				theta,data_dict['features'],data_dict['labels'])

		raise NotImplementedError(f"Statistic: {statistic_name} is not implemented")

	def accuracy(self,model,theta,X,Y):
		prediction = model.predict(theta,X)
		acc = np.mean(1.0*prediction==Y)
		return acc

	def perceptron_loss(self,model,theta,X,Y):
		# print(f"Model={model}")
		prediction = model.predict(theta,X)
		res = np.mean(prediction!=Y) # 0 if all correct, 1 if all incorrect
		return res

	def logistic_loss(self,model,theta,X,Y):
		n = len(X)
		Y_mask = Y==1
		prediction = model.predict(theta,X)
		# res = np.sum(-Y*np.log(prediction) - (1.0-Y)*np.log(1.0-prediction))
		res = np.sum(-np.log(prediction[Y_mask])) # Y==1
		res += np.sum(-np.log(1.0 - prediction[~Y_mask])) # Y==0
		res /=n
		return res

	def sample_Positive_Rate(self,model,theta,X,Y):
		"""
		Calculate positive rate
		for the whole sample.
		This happens when prediction = 1
		Outputs a value between 0 and 1
		"""
		prediction = self.predict(theta,X)
		X_P = X.loc[prediction==1]
		return len(X_P)/len(X)

	def sample_Negative_Rate(self,model,theta,X,Y):
		"""
		Calculate positive rate
		for the whole sample.
		This happens when prediction = 1
		Outputs a value between 0 and 1
		"""
		prediction = self.predict(theta,X)
		X_N = X.loc[prediction!=1]
		return len(X_N)/len(X)

	def sample_False_Positive_Rate(self,model,theta,X,Y):
		"""
		Calculate false positive rate
		for the whole sample.
		This happens when prediction = 1
		and label = 0.
		Outputs a value between 0 and 1
		"""
		prediction = self.predict(theta,X)
		X_FP = X.loc[np.logical_and(Y!=1,prediction==1)]
		return len(X_FP)/len(X)

	def sample_False_Negative_Rate(self,model,theta,X,Y):
		"""
		Calculate false negative rate
		for the whole sample.
		This happens when prediction = 0
		and label = 1.
		Outputs a value between 0 and 1
		"""
		prediction = self.predict(theta,X)
		X_FN = X.loc[np.logical_and(Y==1,prediction!=1)]
		return len(X_FN)/len(X) 

	def Vector_Positive_Rate(self,model,theta,X,Y):
		"""
		Calculate positive rate
		for each observation.
		This happens when prediction = 1
		Outputs a value between 0 and 1
		"""

		prediction = self.predict(theta,X)
		P_mask = prediction==1
		return 1.0*P_mask

	def Vector_Negative_Rate(self,model,theta,X,Y):
		"""
		Calculate negative rate
		for each observation.
		This happens when prediction = -1
		Outputs a value between 0 and 1
		"""
		prediction = self.predict(theta,X)
		N_mask = prediction!=1
		return 1.0*N_mask

	def Vector_False_Positive_Rate(self,model,theta,X,Y):
		"""
		Calculate false positive rate
		for each observation
		This happens when prediction = 1
		and label = -1.
		Outputs a value between 0 and 1
		for each observation
		"""
		prediction = self.predict(theta,X)
		FP_mask = np.logical_and(Y!=1,prediction==1)
		return 1.0*FP_mask

	def Vector_False_Negative_Rate(self,model,theta,X,Y):
		"""
		Calculate false negative rate
		for each observation
		This happens when prediction = -1
		and label = 1.
		Outputs a value between 0 and 1
		for each observation
		"""

		prediction = self.predict(theta,X)
		FN_mask = np.logical_and(Y==1,prediction!=1)
		return 1.0*FN_mask

	def Vector_True_Positive_Rate(self,model,theta,X,Y):
		"""
		Calculate true positive rate
		for each observation
		This happens when prediction = 1
		and label = 1.
		Outputs a value between 0 and 1
		for each observation
		"""
		prediction = self.predict(theta,X)
		TP_mask = np.logical_and(Y==1,prediction==1)
		return 1.0*TP_mask


class LinearClassifierModel(ClassificationModel):
	def __init__(self):
		super().__init__()
		self.model_class = LinearRegression

	def predict(self,theta,X):
		""" Given a set of weights, theta,
		and an array of feature vectors, X, which include offsets
		in the first column,
		make prediction using the model """
		prediction = np.sign(np.dot(theta.T,X.T)) # -1 or 1
		return prediction


class SGDClassifierModel(ClassificationModel):
	def __init__(self):
		super().__init__()
		# self.model_class = LinearRegression
		self.model_class = SGDClassifier

	def fit(self,model,X,Y):
		reg = model.model_class().fit(X, Y)
		return reg.coef_[0]

	def predict(self,theta,X):
		""" Given a set of weights, theta,
		and an array of feature vectors, X, which include offsets
		in the first column,
		make prediction using the model """
		prediction = np.sign(np.dot(theta.T,X.T)) # -1 or 1
		# map -1 to 0
		prediction[prediction==-1]=0
		return prediction


class LogisticRegressionModel(ClassificationModel):
	def __init__(self):
		super().__init__()
		self.model_class = LogisticRegression

	def predict(self,theta,X):
		""" Given a model class instance predict the class label 
		given features X"""
		arg = np.dot(theta.T,X.T)
		val = np.exp(arg)/(1+np.exp(arg))
		prediction = val>=0.5
		# print(prediction)
		return prediction

	def predict_proba(self,theta,X):
		""" Given a model class instance predict the class label 
		given features X"""
		arg = np.dot(theta.T,X.T)
		prediction = np.exp(arg)/(1+np.exp(arg))

		return prediction
		# return model.predict(X.loc[:,X.columns[1:]])

	def fit(self,model,X,Y):
		reg = model.model_class().fit(X, Y)
		return reg.coef_[0]


class RLModel(SeldonianModel):
	def __init__(self):
		pass
		
	def sample_from_statistic(self,
		statistic_name,model,theta,data_dict):
		if statistic_name == 'J_pi_new':
			return model.vector_IS_estimate(model,
				theta,data_dict)

	def IS_estimate(self,model,theta,dataset):
		"""Computed the basic importance sampling weight """
		# For all state, action pairs in data,
		# compute pi_new, the probability of 
		# picking that action at that state
		df = dataset.df

		df['pi_new/pi_b'] = list(map(
			partial(model.apply_policy,theta=theta),
			df['O'],df['A']))/df['pi_b']
		df_new = df.drop(columns=['timestep','O','A','pi_b'])
		g=df_new.groupby('episode_index')
		products_by_episode = g.prod()['pi_new/pi_b']
		result = (
			products_by_episode*g['R'].apply(weighted_sum_gamma)
			).sum()/len(g)
		return result 

	def vector_IS_estimate(self,model,theta,data_dict):
		"""Get an IS estimate vector where
		each entry is the IS estimate from each episode
		"""
		df = data_dict['dataframe']

		df['pi_new/pi_b'] = list(map(
			partial(model.apply_policy,theta=theta),
			df['O'],df['A']))/df['pi_b']
		df_new = df.drop(columns=['timestep','O','A','pi_b'])
		g=df_new.groupby('episode_index')
		products_by_episode = g.prod()['pi_new/pi_b']

		result = (
			products_by_episode*data_dict['reward_sums_by_episode']
			)
		return result 

class SoftmaxRLModel(RLModel):
	def __init__(self,environment):
		self.environment = environment
	
	def apply_policy(self,state,action,theta):
		""" Apply the softmax policy given a state and action
		as well as the set of policy parameters, theta.
		Theta is a flattened parameter vector """
		state = int(state)
		action = int(action)
		index = state*4+action
		# print(index)
		arg = theta[index]
		# print(arg)
		return np.exp(arg)/np.sum(np.exp(theta[state*4+self.environment.actions]))