""" Main module containing Seldonian machine learning models """ 

import autograd.numpy as np   # Thinly-wrapped version of Numpy
from sklearn.linear_model import (LinearRegression,
	LogisticRegression, SGDClassifier)
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
		super().__init__()


class RegressionModel(SupervisedModel):
	def __init__(self):
		""" Parent class for all regression-based machine learning 
		models used in this library """ 
		super().__init__()

	def predict(self):
		raise NotImplementedError("Implement this method in child class")

	def evaluate_statistic(self,
		statistic_name,theta,data_dict):
		""" Evaluate a provided statistic for the whole sample provided

		:param statistic_name: The name of the statistic to evaluate
		:type statistic_name: str, e.g. 'Mean_Squared_Error'

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param data_dict: Contains the features and labels 
		:type data_dict: dict

		:return: The evaluated statistic over the whole sample
		:rtype: float
		"""
		if statistic_name == 'Mean_Squared_Error':
			return self.sample_Mean_Squared_Error(
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'Mean_Error':
			return self.sample_Mean_Error(
				theta,data_dict['features'],data_dict['labels'])

		raise NotImplementedError(f"Statistic: {statistic_name} is not implemented")

	def sample_from_statistic(self,
		statistic_name,theta,data_dict):
		""" Evaluate a provided statistic for each observation 
		in the sample

		:param statistic_name: The name of the statistic to evaluate
		:type statistic_name: str, e.g. 'Mean_Squared_Error'

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param data_dict: Contains the features and labels 
		:type data_dict: dict

		:return: The evaluated statistic for each observation in the sample
		:rtype: numpy ndarray(float)
		"""
		if statistic_name == 'Mean_Squared_Error':
			return self.vector_Mean_Squared_Error(
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'Mean_Error':
			return self.vector_Mean_Error(
				theta,data_dict['features'],data_dict['labels'])

		raise NotImplementedError(f"Statistic: {statistic_name} is not implemented")

	def sample_Mean_Squared_Error(self,theta,X,Y):
		"""
		Calculate sample mean squared error 

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
		prediction = self.predict(theta,X) # vector of values
		res = sum(pow(prediction-Y,2))/n
		return res

	def gradient_sample_Mean_Squared_Error(self,theta,X,Y):
		n = len(X)
		prediction = self.predict(theta,X) # vector of values
		err = prediction-Y
		return 2/n*np.dot(err,X)
	
	def sample_Mean_Error(self,theta,X,Y):
		"""
		Calculate sample mean error 

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
		prediction = self.predict(theta,X) # vector of values
		res = sum(prediction-Y)/n
		return res

	def vector_Mean_Squared_Error(self,theta,X,Y):
		""" Calculate squared error for each observation
		in the dataset

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: vector of mean squared error values
		:rtype: numpy ndarray(float)
		"""  
		prediction = self.predict(theta, X)
		return pow(prediction-Y,2)
		
	def vector_Mean_Error(self,theta,X,Y):
		""" Calculate mean error for each observation
		in the dataset

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: vector of mean error values
		:rtype: numpy ndarray(float)
		"""  
		prediction = self.predict(theta, X)
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

	def default_objective(self,theta,X,Y):
		""" The default primary objective to use, the 
		sample mean squared error

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: Sample mean squared error
		:rtype: float
		"""
		return self.sample_Mean_Squared_Error(theta,X,Y)

	def gradient_default_objective(self,theta,X,Y):
		""" The gradient of the default primary objective to use, the 
		sample mean squared error

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: Gradient of the sample mean squared error 
			evaluated at theta
		:rtype: float
		"""
		return self.gradient_sample_Mean_Squared_Error(theta,X,Y)

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


class SquashedLinearRegressionModel(LinearRegressionModel):
	def __init__(self):
		""" Implements linear regression 
		with a squashed predict function.
		Overrides several parent methods """
		super().__init__()
		self.model_class = LinearRegression

	def sample_Squashed_Squared_Error(self,theta,X,Y):
		"""
		Calculate sample mean squared error 

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
		prediction = self.predict(theta,X,Y) # vector of values
		res = sum(pow(prediction-Y,2))/n
		return res

	def vector_Squashed_Squared_Error(self,theta,X,Y):
		"""
		Calculate vector of squashed squared errors 

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
		prediction = self.predict(theta,X,Y) # vector of values
		res = pow(prediction-Y,2)
		return res

	def gradient_sample_Squashed_Squared_Error(self,theta,X,Y):
		n = len(X)
		# prediction = sigma(Xw)
		prediction = self.predict(theta,X,Y) # vector of values
		err = prediction-Y
		# return 2/n*np.dot(err,prediction)*np.dot(1-prediction,X)
		return 2/n*np.dot(err,prediction)*np.dot(1-prediction,X)

	def _sigmoid(self,X):
		return 1/(1+np.exp(X))

	def predict(self,theta,X,Y):
		""" Overrides the original predict
		function to squash predictions 

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:return: predicted labels
		:rtype: numpy ndarray
		"""
		Y_min,Y_max = min(Y),max(Y)
		# Want range of Y_hat to be twice that of Y
		# and want size of interval on either side of Y_min and Y_max
		# to be the same. The unique solution to this is:
		Y_hat_min = (3*Y_min - Y_max)/2
		Y_hat_max =(3*Y_max - Y_min)/2
		Z = np.dot(X,theta)
		return self._sigmoid(Z)*(Y_hat_max-Y_hat_min) + Y_hat_min


class ClassificationModel(SupervisedModel):
	def __init__(self):
		""" Parent class for all classification-based 
		machine learning models used in this library. 

		Currently only supports binary classification
		"""
		super().__init__()

	def predict(self):
		raise NotImplementedError("Implement this method in child class")
		
	def evaluate_statistic(self,
		statistic_name,theta,data_dict):
		""" Evaluate a provided statistic for the whole sample provided

		:param statistic_name: The name of the statistic to evaluate
		:type statistic_name: str, e.g. 'FPR' for false positive rate

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param data_dict: Contains the features and labels 
		:type data_dict: dict

		:return: The evaluated statistic over the whole sample
		:rtype: float
		"""
		if statistic_name == 'PR':
			return self.sample_Positive_Rate(
				theta,data_dict['features'])

		if statistic_name == 'NR':
			return self.sample_Negative_Rate(
				theta,data_dict['features'])

		if statistic_name == 'FPR':
			return self.sample_False_Positive_Rate(
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'FNR':
			return self.sample_False_Negative_Rate(
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'TPR':
			return self.sample_True_Positive_Rate(
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'TNR':
			return self.sample_True_Negative_Rate(
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'logistic_loss':
			return self.sample_logistic_loss(
				theta,data_dict['features'],data_dict['labels'])

		raise NotImplementedError(f"Statistic: {statistic_name} is not implemented")

	def sample_from_statistic(self,
		statistic_name,theta,data_dict):
		""" Evaluate a provided statistic for each observation 
		in the sample

		:param statistic_name: The name of the statistic to evaluate
		:type statistic_name: str, e.g. 'FPR'

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param data_dict: Contains the features and labels 
		:type data_dict: dict

		:return: The evaluated statistic for each observation in the sample
		:rtype: numpy ndarray(float)
		"""
		if statistic_name == 'PR':
			return self.vector_Positive_Rate(
				theta,data_dict['features'])

		if statistic_name == 'NR':
			return self.vector_Negative_Rate(
				theta,data_dict['features'])

		if statistic_name == 'FPR':
			return self.vector_False_Positive_Rate(
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'FNR':
			return self.vector_False_Negative_Rate(
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'TPR':
			return self.vector_True_Positive_Rate(
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'TNR':
			return self.vector_True_Negative_Rate(
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'logistic_loss':
			return self.vector_logistic_loss(
				theta,data_dict['features'],data_dict['labels'])

		raise NotImplementedError(f"Statistic: {statistic_name} is not implemented")

	def sample_logistic_loss(self,theta,X,Y):
		""" Calculate logistic loss 
		on whole sample

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: logistic loss
		:rtype: float
		"""
		z = np.dot(X,theta[1:]) + theta[0]
		h = 1/(1+np.exp(-z))
		res = np.mean(-Y*np.log(h) - (1.0-Y)*np.log(1.0-h))
		return res

	def vector_logistic_loss(self,theta,X,Y):
		""" Calculate logistic loss 
		on each observation in sample

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: array of logistic losses 
		:rtype: numpy ndarray(float)
		"""
		z = np.dot(X,theta[1:]) + theta[0]
		h = 1/(1+np.exp(-z))
		res = -Y*np.log(h) - (1.0-Y)*np.log(1.0-h)
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
		
		z = np.dot(X,theta[1:]) + theta[0]
		h = 1/(1+np.exp(-z))
		X_withintercept = np.hstack([np.ones((len(X),1)),np.array(X)])
		res = (1/len(X))*np.dot(X_withintercept.T, (h - Y))
		return res

	def sample_weighted_loss(self,theta,X,Y):
		""" Calculate the averaged weighted cost: 
		sum_i p_(wrong answer for point I) * c_i
		where c_i is 1 for false positives and 5 for false negatives

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: weighted loss such that false negatives 
			have 5 times the cost as false positives
		:rtype: float
		"""
		# calculate probabilistic false positive rate and false negative rate
		y_pred = self.predict(theta,X)
		n_points = len(Y)
		neg_mask = Y!=1 # this includes false positives and true negatives
		pos_mask = Y==1 # this includes true positives and false negatives
		fp_values = y_pred[neg_mask] # get just false positives
		fn_values = 1.0-y_pred[pos_mask] # get just false negatives
		fpr = 1.0*np.sum(fp_values)
		fnr = 5.0*np.sum(fn_values)
		return (fpr + fnr)/n_points

	def vector_weighted_loss(self,theta,X,Y):
		""" Calculate the averaged weighted cost
		on each observation in sample

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: array of weighted losses
		:rtype: numpy ndarray(float)
		"""
		# calculate probabilistic false positive rate and false negative rate
		y_pred = self.predict(theta,X)
		fp_mask = np.logical_and(Y!=1,y_pred==1)
		fn_mask = np.logical_and(Y==1,y_pred!=1)
		# calculate probabilistic false positive rate and false negative rate
		res = np.zeros_like(Y)
		res[fp_mask] = 1.0
		res[fn_mask] = 5.0
		return res

	def sample_Positive_Rate(self,theta,X):
		"""
		Calculate positive rate
		for the whole sample.
		This is the sum of probability of each 
		sample being in the positive class
		normalized to the number of predictions 

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:return: Positive rate for whole sample
		:rtype: float between 0 and 1
		"""	
		prediction = self.predict(theta,X)
		return np.sum(prediction)/len(X) # if all 1s then PR=1. 

	def sample_Negative_Rate(self,theta,X):
		"""
		Calculate negative rate
		for the whole sample.
		This is the sum of the probability of each 
		sample being in the negative class, which is
		1.0 - probability of being in positive class

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:return: Negative rate for whole sample
		:rtype: float between 0 and 1
		"""
		prediction = self.predict(theta,X)
		return np.sum(1.0-prediction)/len(X) # if all 1s then PR=1. 

	def sample_False_Positive_Rate(self,theta,X,Y):
		"""
		Calculate false positive rate
		for the whole sample.
		
		The is the sum of the probability of each 
		sample being in the positive class when in fact it was in 
		the negative class.

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

	def sample_False_Negative_Rate(self,theta,X,Y):
		"""
		Calculate false negative rate
		for the whole sample.
		
		The is the sum of the probability of each 
		sample being in the negative class when in fact it was in 
		the positive class.

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

	def sample_True_Positive_Rate(self,theta,X,Y):
		"""
		Calculate true positive rate
		for the whole sample.
		
		The is the sum of the probability of each 
		sample being in the positive class when in fact it was in 
		the positive class.

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
		pos_mask = Y==1.0 # this includes false positives and true negatives
		return np.sum(prediction[pos_mask])/len(X[pos_mask])

	def sample_True_Negative_Rate(self,theta,X,Y):
		"""
		Calculate true negative rate
		for the whole sample.
		
		The is the sum of the probability of each 
		sample being in the negative class when in fact it was in 
		the negative class.

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:return: False positive rate for whole sample
		:rtype: float between 0 and 1
		"""
		prediction = self.predict(theta,X)
		# Sum the probability of being in negative class
		# subject to the truth being the negative class
		neg_mask = Y!=1.0 # this includes false positives and true negatives
		return np.sum(1.0-prediction[neg_mask])/len(X[neg_mask])
		
	def vector_Positive_Rate(self,theta,X):
		"""
		Calculate positive rate
		for each observation.
		
		This is the probability of being positive

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

	def vector_Negative_Rate(self,theta,X):
		"""
		Calculate negative rate
		for each observation.

		This is the probability of being negative

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:return: Positive rate for each observation
		:rtype: numpy ndarray(float between 0 and 1)
		"""
		prediction = self.predict(theta,X)

		return 1.0 - prediction

	def vector_False_Positive_Rate(self,theta,X,Y):
		"""
		Calculate false positive rate
		for each observation

		This is the probability of predicting positive
		subject to the label actually being negative
	

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

	def vector_False_Negative_Rate(self,theta,X,Y):
		"""
		Calculate false negative rate
		for each observation
		
		This is the probability of predicting negative
		subject to the label actually being positive

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

	def vector_True_Positive_Rate(self,theta,X,Y):
		"""
		This is the probability of predicting positive
		subject to the label actually being positive

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

	def vector_True_Negative_Rate(self,theta,X,Y):
		"""
		This is the probability of predicting negative
		subject to the label actually being negative

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

	def default_objective(self,theta,X,Y):
		""" The default primary objective to use, the 
		logistic loss

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:param Y: The labels
		:type Y: numpy ndarray

		:return: Logistic loss
		:rtype: float
		"""
		return self.sample_logistic_loss(theta,X,Y)

	def gradient_default_objective(self,theta,X,Y):
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
		return self.gradient_sample_logistic_loss(theta,X,Y)


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
		z = np.dot(X,theta[1:]) + theta[0]
		h = 1/(1+np.exp(-z))

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
		# self.predictor = LogisticRegression()
		reg = self.model_class().fit(X, Y)
		return np.squeeze(np.hstack([reg.intercept_.reshape(-1,1),reg.coef_]))


class DummyClassifierModel(ClassificationModel):
	def __init__(self):
		""" Implements a classifier that always predicts
		the positive class, regardless of input """
		super().__init__()
		self.model_class = None

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

		return np.ones(len(X))


class RandomClassifierModel(ClassificationModel):
	def __init__(self):
		""" Implements a classifier that always predicts
		that the positive class has prob=0.5,
		regardless of input """
		super().__init__()
		self.model_class = None

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

		return 0.5*np.ones(len(X))

