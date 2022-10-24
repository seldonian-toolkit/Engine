""" Main module containing Seldonian machine learning models """ 

import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd.extend import primitive, defvjp
from sklearn.linear_model import (LinearRegression,
	LogisticRegression, SGDClassifier)
from .models import RegressionModel
from functools import partial, lru_cache

import torch


class PytorchLinearRegressionModel2():
	def __init__(self,input_dim,output_dim):
		""" Implements linear regression """
		super().__init__()
		self.linear = torch.nn.Linear(input_dim, output_dim)
		self.has_intercept=True

	def predict(self,theta,X):
		""" Predict label using the linear model

		:param theta: The parameter weights
		:type theta: numpy ndarray
		:param X: The features
		:type X: numpy ndarray
		:return: predicted labels
		:rtype: numpy ndarray
		"""
		# Cant have any pytorch in here

		return mypredict(theta,X,self)


@primitive
def mypredict(theta,X,model):
	# Can have pytorch stuff in here but cant explicitly return a pytorch object
	# First update model weights
	slope_torch = torch.tensor(theta[1:],requires_grad=True).view(-1,1)
	with torch.no_grad():
		model.linear.bias[0] = theta[0]
		model.linear.weight[:] = torch.tensor(theta[1:])
	global pred
	X_torch = torch.tensor(X,requires_grad=True)
	pred = model.linear(X_torch.float()).view(-1)
	pred_numpy = pred.detach().numpy()
	return pred_numpy


def mypredict_vjp(ans,theta,X,model):
#     print(len(X))
	external_grad = torch.ones(len(X)) # dQ/dQ
	def fn(v):
		# v is a unit vector of shape ans, the return value of mypredict()
		# return a 1D array [dF_i/dtheta[0],dF_i/dtheta[1],dF_i/dtheta[2]],
		# where i is the data row index
		if model.linear.bias.grad is not None:
			model.linear.bias.grad.zero_()
		if model.linear.weight.grad is not None:
			model.linear.weight.grad.zero_()
		external_grad = torch.tensor(v)
		pred.backward(gradient=external_grad,retain_graph=True)
		grad = torch.cat((model.linear.bias.grad,model.linear.weight.grad.view(-1)))
		return np.array(grad)
	return fn

	
defvjp(mypredict,mypredict_vjp)
