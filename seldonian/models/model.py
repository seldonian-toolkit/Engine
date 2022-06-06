""" Main module containing Seldonian machine learning models """ 

import autograd.numpy as np   # Thinly-wrapped version of Numpy
from sklearn.linear_model import (LinearRegression,
	LogisticRegression, SGDClassifier)
from seldonian.utils.stats_utils import weighted_sum_gamma
from functools import partial, lru_cache

class SeldonianModel(object):
	def __init__(self):
		""" Parent class for all machine learning models
		used in this library. """
		pass


class SupervisedModel(SeldonianModel):
	def __init__(self):
		""" Parent class for all supervised machine learning 
		models used in this library """
		pass

	def fit(self,X,Y):
		""" Train the model using the feature,label pairs 

		:param X: features 
		:type X: NxM numpy ndarray 

		:param X: labels 
		:type X: Nx1 numpy ndarray 

		:return: weights from the fitted model
		:rtype: numpy ndarray
		"""
		reg = self.model_class().fit(X, Y)
		return np.hstack([np.array(reg.intercept_),reg.coef_[1:]])


class RegressionModel(SupervisedModel):
	def __init__(self):
		""" Parent class for all regression-based machine learning 
		models used in this library """ 
		super().__init__()

	def evaluate_statistic(self,
		statistic_name,model,theta,data_dict):
		""" Evaluate a provided statistic for the whole sample provided

		:param statistic_name: The name of the statistic to evaluate
		:type statistic_name: str, e.g. 'Mean_Squared_Error'

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param data_dict: Contains the features and labels 
		:type data_dict: dict

		:return: The evaluated statistic over the whole sample
		:rtype: float
		"""
		if statistic_name == 'Mean_Squared_Error':
			return model.sample_Mean_Squared_Error(model,
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'Mean_Error':
			return model.sample_Mean_Error(model,
				theta,data_dict['features'],data_dict['labels'])

		raise NotImplementedError(f"Statistic: {statistic_name} is not implemented")

	def sample_from_statistic(self,
		statistic_name,model,theta,data_dict):
		""" Evaluate a provided statistic for each observation 
		in the sample

		:param statistic_name: The name of the statistic to evaluate
		:type statistic_name: str, e.g. 'Mean_Squared_Error'

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param data_dict: Contains the features and labels 
		:type data_dict: dict

		:return: The evaluated statistic for each observation in the sample
		:rtype: numpy ndarray(float)
		"""
		if statistic_name == 'Mean_Squared_Error':
			return model.vector_Mean_Squared_Error(model,
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'Mean_Error':
			return model.vector_Mean_Error(model,
				theta,data_dict['features'],data_dict['labels'])

		raise NotImplementedError(f"Statistic: {statistic_name} is not implemented")

	def sample_Mean_Squared_Error(self,model,theta,X,Y):
		"""
		Calculate sample mean squared error 

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: Sample mean squared error
		:rtype: float
		"""
		n = len(X)
		prediction = model.predict(theta,X) # vector of values
		res = sum(pow(prediction-Y,2))/n
		return res

	def sample_Mean_Error(self,model,theta,X,Y):
		"""
		Calculate sample mean error 

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: Sample mean error
		:rtype: float
		"""
		n = len(X)
		prediction = model.predict(theta,X) # vector of values
		res = sum(prediction-Y)/n
		return res

	def vector_Mean_Squared_Error(self,model,theta,X,Y):
		""" Calculate mean squared error for each observation
		in the dataset

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: vector of mean squared error values
		:rtype: numpy ndarray(float)
		"""  
		prediction = model.predict(theta, X)
		return pow(prediction-Y,2)
		
	def vector_Mean_Error(self,model,theta,X,Y):
		""" Calculate mean error for each observation
		in the dataset

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: vector of mean error values
		:rtype: numpy ndarray(float)
		"""  
		prediction = model.predict(theta, X)
		return prediction-Y


class LinearRegressionModel(RegressionModel):
	def __init__(self):
		""" Implements linear regression """
		super().__init__()
		self.model_class = LinearRegression

	def predict(self,theta,X):
		""" Predict label using the linear model

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:return: predicted labels
		:rtype: numpy ndarray
		"""
		return np.dot(X,theta)

	def default_objective(self,model,theta,X,Y):
		""" The default primary objective to use, the 
		sample mean squared error

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: Sample mean squared error
		:rtype: float
		"""
		return self.sample_Mean_Squared_Error(model,theta,X,Y)


class ClassificationModel(SupervisedModel):
	def __init__(self):
		""" Parent class for all classification-based 
		machine learning models used in this library. 

		Currently only supports binary classification
		"""
		super().__init__()

	def evaluate_statistic(self,
		statistic_name,model,theta,data_dict):
		""" Evaluate a provided statistic for the whole sample provided

		:param statistic_name: The name of the statistic to evaluate
		:type statistic_name: str, e.g. 'FPR' for false positive rate

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param data_dict: Contains the features and labels 
		:type data_dict: dict

		:return: The evaluated statistic over the whole sample
		:rtype: float
		"""
		if statistic_name == 'PR':
			return model.sample_Positive_Rate(model,
				theta,data_dict['features'])

		if statistic_name == 'NR':
			return model.sample_Negative_Rate(model,
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'FPR':
			return model.sample_False_Positive_Rate(model,
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'FNR':
			return model.sample_False_Negative_Rate(model,
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'TPR':
			return model.sample_True_Positive_Rate(model,
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'logistic_loss':
			return model.sample_logistic_loss(model,
				theta,data_dict['features'],data_dict['labels'])

		raise NotImplementedError(f"Statistic: {statistic_name} is not implemented")

	def sample_from_statistic(self,
		statistic_name,model,theta,data_dict):
		""" Evaluate a provided statistic for each observation 
		in the sample

		:param statistic_name: The name of the statistic to evaluate
		:type statistic_name: str, e.g. 'FPR'

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param data_dict: Contains the features and labels 
		:type data_dict: dict

		:return: The evaluated statistic for each observation in the sample
		:rtype: numpy ndarray(float)
		"""
		if statistic_name == 'PR':
			return model.Vector_Positive_Rate(model,
				theta,data_dict['features'])

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
		if statistic_name == 'logistic_loss':
			return model.Vector_logistic_loss(model,
				theta,data_dict['features'],data_dict['labels'])

		raise NotImplementedError(f"Statistic: {statistic_name} is not implemented")

	def accuracy(self,model,theta,X,Y):
		""" Calculate the accuracy of the 
		binary classification model 

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: accuracy
		:rtype: float
		"""
		prediction = model.predict(theta,X)
		predict_class = prediction>=0.5
		acc = np.mean(1.0*predict_class==Y)
		return acc

	def sample_perceptron_loss(self,model,theta,X,Y):
		""" Calculate perceptron loss 
		(fraction of incorrect classifications) on whole sample

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: perceptron loss
		:rtype: float
		"""
		prediction = model.predict(theta,X)
		res = np.mean(prediction!=Y) # 0 if all correct, 1 if all incorrect
		return res

	def sample_logistic_loss(self,model,theta,X,Y):
		""" Calculate logistic loss 
		on whole sample

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: logistic loss
		:rtype: float
		"""
		h = 1/(1+np.exp(-1.0*np.dot(X,theta)))
		res = np.mean(-Y*np.log(h) - (1.0-Y)*np.log(1.0-h))
		return res

	def gradient_sample_logistic_loss(self,theta,X,Y):
		""" Gradient of logistic loss w.r.t. theta

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: perceptron loss
		:rtype: float
		"""
		h = 1/(1+np.exp(-1.0*np.dot(X,theta)))
		return (1/len(X))*np.dot(X.T, (h - Y))

	def Vector_logistic_loss(self,model,theta,X,Y):
		""" Calculate logistic loss 
		on each observation in sample

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: array of logistic losses for each observation
		:rtype: numpy ndarray(float)
		"""
		h = 1/(1+np.exp(-1.0*np.dot(X,theta)))
		res = -Y*np.log(h) - (1.0-Y)*np.log(1.0-h)
		return res		

	def sample_Positive_Rate(self,model,theta,X):
		"""
		Calculate positive rate
		for the whole sample.
		This is the sum of probability of each 
		sample being in the positive class
		normalized to the number of predictions 

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:return: Positive rate for whole sample
		:rtype: float between 0 and 1
		"""
		prediction = self.predict(theta,X)
		return np.sum(prediction)/len(X) # if all 1s then PR=1. 

	def sample_Negative_Rate(self,model,theta,X):
		"""
		Calculate negative rate
		for the whole sample.
		This is the sum of the probability of each 
		sample being in the negative class, which is
		1.0 - probability of being in positive class

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:return: Negative rate for whole sample
		:rtype: float between 0 and 1
		"""
		prediction = self.predict(theta,X)
		return np.sum(1.0-prediction)/len(X) # if all 1s then PR=1. 

	def sample_False_Positive_Rate(self,model,theta,X,Y):
		"""
		Calculate false positive rate
		for the whole sample.
		
		The is the sum of the probability of each 
		sample being in the positive class when in fact it was in 
		the negative class.

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:return: False positive rate for whole sample
		:rtype: float between 0 and 1
		"""
		prediction = self.predict(theta,X)
		# Sum the probability of being in positive class
		# subject to the truth being the other class
		neg_mask = Y!=1.0 # this includes false positives and true negatives
		return np.sum(prediction[neg_mask])/len(X[neg_mask])

	def sample_False_Negative_Rate(self,model,theta,X,Y):
		"""
		Calculate false negative rate
		for the whole sample.
		
		The is the sum of the probability of each 
		sample being in the negative class when in fact it was in 
		the positive class.

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:return: False negative rate for whole sample
		:rtype: float between 0 and 1
		"""
		prediction = self.predict(theta,X)
		# Sum the probability of being in negative class
		# subject to the truth being the positive class
		pos_mask = Y==1.0 # this includes false positives and true negatives
		return np.sum(1.0-prediction[pos_mask])/len(X[pos_mask])

	def Vector_Positive_Rate(self,model,theta,X):
		"""
		Calculate positive rate
		for each observation.
		
		This is the probability of being positive

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:return: Positive rate for each observation
		:rtype: numpy ndarray(float between 0 and 1)
		"""
		# prediction = self.predict(theta,X)
		# P_mask = prediction==1.0
		# res = 1.0*P_mask
		# return 1.0*P_mask
		prediction = self.predict(theta,X) # probability of class 1 for each observation
		return prediction 

	def Vector_Negative_Rate(self,model,theta,X):
		"""
		Calculate negative rate
		for each observation.

		This is the probability of being negative

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:return: Positive rate for each observation
		:rtype: numpy ndarray(float between 0 and 1)
		"""
		prediction = self.predict(theta,X)

		return 1.0 - prediction

	def Vector_False_Positive_Rate(self,model,theta,X,Y):
		"""
		Calculate false positive rate
		for each observation

		This is the probability of predicting positive
		subject to the label actually being negative
		
		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:return: False positive rate for each observation
		:rtype: numpy ndarray(float between 0 and 1)
		"""
		prediction = self.predict(theta,X)
		# The probability of being in positive class
		# subject to the truth being the other class
		neg_mask = Y!=1.0 # this includes false positives and true negatives
		return prediction[neg_mask]

	def Vector_False_Negative_Rate(self,model,theta,X,Y):
		"""
		Calculate false negative rate
		for each observation
		
		This is the probability of predicting negative
		subject to the label actually being positive

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:return: False negative rate for each observation
		:rtype: numpy ndarray(float between 0 and 1)
		"""

		prediction = self.predict(theta,X)
		# The probability of being in positive class
		# subject to the truth being the other class
		pos_mask = Y==1.0 # this includes false positives and true negatives
		return 1.0-prediction[pos_mask]

	def Vector_True_Positive_Rate(self,model,theta,X,Y):
		"""
		This is the probability of predicting positive
		subject to the label actually being positive

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:return: True positive rate for each observation
		:rtype: numpy ndarray(float between 0 and 1)
		"""
		prediction = self.predict(theta,X)
		pos_mask = Y==1.0 # this includes false positives and true negatives
		return prediction[pos_mask]

	def Vector_True_Negative_Rate(self,model,theta,X,Y):
		"""
		This is the probability of predicting negative
		subject to the label actually being negative

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:return: True negative rate for each observation
		:rtype: numpy ndarray(float between 0 and 1)
		"""
		prediction = self.predict(theta,X)
		pos_mask = Y!=1.0 # this includes false positives and true negatives
		return 1.0 - prediction[pos_mask]


class LinearClassifierModel(ClassificationModel):
	def __init__(self):
		""" Implements a linear classifier using linear regression"""
		super().__init__()
		self.model_class = LinearRegression

	def predict(self,theta,X):
		""" Make prediction using the model weights

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:return: predictions for each observation
		:rtype: numpy ndarray(float)
		"""
		prediction = np.sign(np.dot(X,theta)) # -1 or 1
		return prediction

	def default_objective(self,model,theta,X,Y):
		""" The default primary objective to use, the 
		perceptron loss

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: Perceptron loss
		:rtype: float
		"""
		return self.sample_perceptron_loss(model,theta,X,Y)


class SGDClassifierModel(ClassificationModel):
	def __init__(self):
		""" Implements a linear support vector machine """
		super().__init__()
		# self.model_class = LinearRegression
		self.model_class = SGDClassifier

	def fit(self,X,Y):
		reg = self.model_class().fit(X, Y)
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
		""" Implements logistic regression """
		super().__init__()
		self.model_class = LogisticRegression

	def predict(self,theta,X):
		""" Predict the probability of 
		having the positive class label

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:return: predictions for each observation
		:rtype: float
		"""
		h = 1/(1+np.exp(-1.0*np.dot(X,theta)))

		return h

	def fit(self,X,Y):
		""" Train the model using features and labels

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: fitted model weights
		:rtype: numpy ndarray(float)
		"""
		reg = self.model_class().fit(X, Y)
		return reg.coef_[0]

	def default_objective(self,model,theta,X,Y):
		""" The default primary objective to use, the 
		logistic loss

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: Logistic loss
		:rtype: float
		"""
		return self.sample_logistic_loss(model,theta,X,Y)


class RLModel(SeldonianModel):
	def __init__(self):
		""" The base class for all RL Seldonian models"""
		pass
		
	def sample_from_statistic(self,
		statistic_name,model,theta,data_dict):
		""" Evaluate a provided statistic for each episode 
		in the sample

		:param statistic_name: The name of the statistic to evaluate
		:type statistic_name: str, e.g. 'J_pi_new'

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param data_dict: Contains the dataframe 
		:type data_dict: dict

		:return: The evaluated statistic for each episode in the dataset
		:rtype: numpy ndarray(float)
		"""

		if statistic_name == 'J_pi_new':
			return model.vector_IS_estimate(model,
				theta,data_dict)
		else:
			raise NotImplementedError(
				f"Statistic: {statistic_name} is not implemented")

	def evaluate_statistic(self,
		statistic_name,model,theta,data_dict):
		""" Evaluate a provided statistic for the whole sample provided

		:param statistic_name: The name of the statistic to evaluate
		:type statistic_name: str, e.g. 'J_pi_new'

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param data_dict: Contains the dataframe
		:type data_dict: dict

		:return: The evaluated statistic over the whole sample
		:rtype: float
		"""
		if statistic_name == 'J_pi_new':
			return model.sample_IS_estimate(model,
				theta,data_dict['dataframe'])
		else:
			raise NotImplementedError(
				f"Statistic: {statistic_name} is not implemented")

	def sample_IS_estimate(self,model,theta,dataframe):
		""" Calculate the unweighted importance sampling estimate
		on all episodes in the dataframe

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param dataframe: Contains the episodes
		:type dataframe: pandas dataframe

		:return: The IS estimate calculated over all episodes
		:rtype: float
		"""

		"""Set instance variable theta
		does this so a cache can be used 
		to accelerate the apply_policy() computation 
		on all state,action pairs """
		model.theta = theta
		pi_ratios = list(map(model.apply_policy,
					dataframe['O'].values,
					dataframe['A'].values))/dataframe['pi'].values
		# clear cache so it is safe to use for next time
		model.theta = None 
		model.denom.cache_clear()
		model.arg.cache_clear()
		split_indices_by_episode = np.unique(dataframe['episode_index'].values,
			return_index=True)[1][1:]
		pi_ratios_by_episode = np.split(pi_ratios, split_indices_by_episode) # this is a list
		products_by_episode = np.array(list(map(np.prod,pi_ratios_by_episode)))
		
		# Weighted rewards
		gamma = self.environment.gamma
		rewards_by_episode = np.split(dataframe['R'].values,split_indices_by_episode)
		weighted_reward_sums = np.array(list(map(weighted_sum_gamma,
			rewards_by_episode,gamma*np.ones_like(rewards_by_episode))))
		result = sum(products_by_episode*weighted_reward_sums)/len(pi_ratios_by_episode)
		return result 

	def vector_IS_estimate(self,model,theta,data_dict):
		""" Calculate the unweighted importance sampling estimate
		on each episodes in the dataframe

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param dataframe: Contains the episodes
		:type dataframe: pandas dataframe

		:return: A vector of IS estimates calculated for each episode
		:rtype: numpy ndarray(float)
		"""

		"""set instance variable theta
		does this so a cache can be used 
		to accelerate the apply_policy() computation 
		on all state,action pairs """
		model.theta = theta
		pi_ratios = list(map(
			model.apply_policy,
					data_dict['dataframe']['O'].values,
					data_dict['dataframe']['A'].values
					))/data_dict['dataframe']['pi'].values

		# clear cache so it is safe to use for next time
		model.theta = None
		model.denom.cache_clear()
		model.arg.cache_clear()

		split_indices_by_episode = np.unique(
			data_dict['dataframe']['episode_index'].values,
			return_index=True)[1][1:]
		pi_ratios_by_episode = np.split(pi_ratios, split_indices_by_episode) # this is a list
		products_by_episode = np.array(list(map(np.prod,pi_ratios_by_episode)))
		result = (
			products_by_episode*data_dict['reward_sums_by_episode']
			)
		return result

class TabularSoftmaxModel(RLModel):
	def __init__(self,environment):
		""" Tabular softmax model used for e.g. gridworld 
		environment

		:param environment: the RL environment object
		:type environment: Environment object from RL 
			environment module

		:ivar theta: The model weights
		:vartype theta: numpy ndarray
		"""
		self.environment = environment
		self.theta = None

	@lru_cache
	def _denom(self,state):
		""" Helper function for apply_policy()
		
		:param state: A state in the environment
		:type state: int
		"""
		return np.sum(np.exp(self.theta[state*4+self.environment.actions]))

	@lru_cache
	def _arg(self,state,action):
		""" Helper function for apply_policy()
		
		:param state: A state in the environment
		:type state: int

		:param action: A possible action at the given state
		:type action: int
		"""
		return self.theta[state*4+action]

	def apply_policy(self,state,action):
		""" Apply the softmax policy given a state and action.
		Uses self.theta for the policy parameters in helper functions

		:param state: A state in the environment
		:type state: int

		:param action: A possible action at the given state
		:type action: int
		"""
		state = int(state)
		action = int(action)
		
		return np.exp(self._arg(state,action))/self._denom(state)

	def default_objective(self,model,theta,dataset):
		""" The default primary objective to use, the 
		unweighted IS estimate

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param dataset: The object containing data and metadata
		:type dataset: dataset.Dataset object
		"""
		return self.IS_estimate(model,theta,dataset)


class LinearSoftmaxModel(RLModel):
	def __init__(self,environment):
		""" Linear softmax model used for e.g. mountaincar
		environment

		:param environment: the RL environment object
		:type environment: Environment object from 
			the RL environment module  

		:ivar theta: The model weights
		:vartype theta: numpy ndarray
		"""
		self.environment = environment
		self.theta = None

	def IS_estimate(self,model,theta,dataset):
		""" Calculate the unweighted importance sampling estimate
		on all episodes in the dataframe. Overrides parent method
		so that it can normalizes returns to [0,1].

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param dataset: Contains the dataframe and metadata
		:type dataset: :py:class:`.DataSet` object

		:return: The IS estimate calculated over all episodes
		:rtype: float
		"""

		self.theta = theta
		pi_ratios = list(map(self.apply_policy,
					dataset.df['O'].values,
					dataset.df['A'].values))/dataset.df['pi'].values
		self.theta = None
		split_indices_by_episode = np.unique(dataset.df['episode_index'].values,
			return_index=True)[1][1:]
		pi_ratios_by_episode = np.split(pi_ratios, split_indices_by_episode) # this is a list
		products_by_episode = np.array(list(map(np.prod,pi_ratios_by_episode)))
		
		# Weighted rewards
		gamma = self.environment.gamma
		rewards_by_episode = np.split(dataset.df['R'].values,
			split_indices_by_episode)
		weighted_reward_sums = np.array(list(map(weighted_sum_gamma,
			rewards_by_episode,gamma*np.ones_like(rewards_by_episode))))
		
		# normalize to [0,1]
		min_return = self.environment.min_return
		max_return = self.environment.max_return
		normalized_returns = (weighted_reward_sums-min_return)/(max_return-min_return)

		result = sum(products_by_episode*normalized_returns)/len(pi_ratios_by_episode)
		return result 

	def vector_IS_estimate(self,model,theta,data_dict):
		""" Calculate the unweighted importance sampling estimate
		on each episode in the dataframe. Overrides parent method.

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param dataset: Contains the dataframe and metadata
		:type dataset: :py:class:`.DataSet` object

		:return: The IS estimates calculated on each episode
		:rtype: numpy ndarray(float)
		"""
		self.theta = theta
		pi_ratios = list(map(self.apply_policy,
					data_dict['dataframe']['O'].values,
					data_dict['dataframe']['A'].values))/data_dict['dataframe']['pi'].values
		self.theta = None
		split_indices_by_episode = np.unique(data_dict['dataframe']['episode_index'].values,
			return_index=True)[1][1:]
		pi_ratios_by_episode = np.split(pi_ratios, split_indices_by_episode) # this is a list
		products_by_episode = np.array(list(map(np.prod,pi_ratios_by_episode)))
		result = (
			products_by_episode*data_dict['reward_sums_by_episode']
			)
		return result

	def apply_policy(self, state, action)->float:
		""" Get the probability of taking action in given state 
		
		:param state: A state in the environment
		:type state: numpy ndarray

		:param action: A possible action at the given state
		:type action: int

		:return: Probability of taking action in given state
		:rtype: float
		"""
		x = self.environment.basis.encode(state)
		p = self.get_p(x)
		return p[action]

	def get_p(self, x):
		""" Get vector of probabilites given encoded state, x

		:param x: Encoded state
		:type x: numpy ndarray

		:return: A vector of probablities, one for each 
			possible action in given encoded state
		:rtype: numpy ndarray
		""" 
		u = np.exp(np.clip(np.dot(x, 
			self.theta.reshape(self.environment.policy.n_inputs, self.environment.policy.n_actions)), -32, 32)) 
		u /= u.sum()

		return u

	def default_objective(self,model,theta,dataset):
		""" The default primary objective to use, the 
		IS estimate defined in this class

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param dataset: Contains the dataframe and metadata
		:type dataset: :py:class:`.DataSet` object

		:return: IS estimate for whole sample
		:rtype: float
		"""
		return self.IS_estimate(model,theta,dataset)
