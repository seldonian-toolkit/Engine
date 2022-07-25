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

		:param Y: labels 
		:type Y: Nx1 numpy ndarray 

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

	def gradient_sample_Mean_Squared_Error(self,model,theta,X,Y):
		n = len(X)
		prediction = model.predict(theta,X) # vector of values
		err = prediction-Y
		return 2/n*np.dot(err,X)
		
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
			return model.vector_Positive_Rate(model,
				theta,data_dict['features'])

		if statistic_name == 'NR':
			return model.vector_Negative_Rate(model,
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'FPR':
			return model.vector_False_Positive_Rate(model,
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'FNR':
			return model.vector_False_Negative_Rate(model,
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'TPR':
			return model.vector_True_Positive_Rate(model,
				theta,data_dict['features'],data_dict['labels'])
		if statistic_name == 'logistic_loss':
			return model.vector_logistic_loss(model,
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

	def gradient_sample_logistic_loss(self,model,theta,X,Y):
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

	def vector_logistic_loss(self,model,theta,X,Y):
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

	def vector_Positive_Rate(self,model,theta,X):
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

	def vector_Negative_Rate(self,model,theta,X):
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

	def vector_False_Positive_Rate(self,model,theta,X,Y):
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

	def vector_False_Negative_Rate(self,model,theta,X,Y):
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

	def vector_True_Positive_Rate(self,model,theta,X,Y):
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

	def vector_True_Negative_Rate(self,model,theta,X,Y):
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


class RLEvaluator():
	def __init__(self):
		""" The base class for all RL evaluators"""
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
			return self.vector_IS_estimate(model,
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
			return self.sample_IS_estimate(model,
				theta,data_dict)
		else:
			raise NotImplementedError(
				f"Statistic: {statistic_name} is not implemented")

	def sample_IS_estimate(self, model, theta, data_dict):
		""" Calculate the unweighted importance sampling estimate
		on all episodes in the dataframe

		:param model: The Seldonian model object
		:type model: :py:class:`.SeldonianModel` object

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param dataset: The object containing data and metadata
		:type dataset: dataset.Dataset object

		:return: The IS estimate calculated over all episodes
		:rtype: float
		"""
		episodes = data_dict['episodes']
		IS_estimate = 0
		for ii, ep in enumerate(episodes):
			pi_news = model.get_probs_from_observations_and_actions(theta, ep.states, ep.actions)
			# print(pi_news,ep.pis)
			pi_ratios = pi_news / ep.pis
			# print(pi_ratios)
			pi_ratio_prod = np.prod(pi_ratios)
			# print(pi_ratio_prod)
			weighted_return = weighted_sum_gamma(ep.rewards, gamma=model.env.gamma)
			# print(weighted_return)
			IS_estimate += pi_ratio_prod * weighted_return

		IS_estimate /= len(episodes)

		return IS_estimate

	def vector_IS_estimate(self, model, theta, data_dict):
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
		episodes = data_dict['episodes']
		# weighted_reward_sums_by_episode = data_dict['reward_sums_by_episode']
		result = []
		for ii, ep in enumerate(episodes):
			pi_news = model.get_probs_from_observations_and_actions(theta, ep.states, ep.actions)
			# print("pi news:")
			# print(pi_news)
			pi_ratio_prod = np.prod(pi_news / ep.pis)
			# print("pi_ratio_prod:")
			# print(pi_ratio_prod)
			weighted_return = weighted_sum_gamma(ep.rewards, gamma=model.env.gamma)
			# result.append(pi_ratio_prod*weighted_reward_sums_by_episode[ii])
			result.append(pi_ratio_prod * weighted_return)

		# print("\nnp.array(result):")
		# print(np.array(result))
		return np.array(result)
