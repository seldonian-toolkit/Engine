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


class BoundedLinearRegressionModel(LinearRegressionModel):
	def __init__(self):
		""" Implements linear regression 
		with a bounded predict function.
		Overrides several parent methods.
		Assumes y-intercept is 0. """
		super().__init__()

	def _sigmoid(self,X):
		return 1/(1+np.exp(-X))

	def predict(self,theta,X):
		""" Overrides the original predict
		function to squash predictions 

		:param theta: The parameter weights
		:type theta: numpy ndarray

		:param X: The features
		:type X: numpy ndarray

		:return: predicted labels
		:rtype: numpy ndarray
		"""
		y_min,y_max = -3,3
		# Want range of Y_hat to be twice that of Y
		# and want size of interval on either side of Y_min and Y_max
		# to be the same. The unique solution to this is:
		s=2.0 # 1 gives you the same bound size as y
		y_hat_min = y_min*(1+s)/2 + y_max*(1-s)/2
		y_hat_max = y_max*(1+s)/2 + y_min*(1-s)/2
		Z = np.dot(X,theta)
		return self._sigmoid(Z)*(y_hat_max-y_hat_min) + y_hat_min


class ClassificationModel(SupervisedModel):
	def __init__(self):
		""" Parent class for all classification-based 
		machine learning models used in this library. 

		Currently only supports binary classification
		"""
		super().__init__()

	def predict(self):
		raise NotImplementedError("Implement this method in child class")
	

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

