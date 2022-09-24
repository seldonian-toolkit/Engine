""" Objective functions """

import autograd.numpy as np   # Thinly-wrapped version of Numpy
from seldonian.utils.stats_utils import weighted_sum_gamma

def sample_from_statistic(model,
	statistic_name,theta,data_dict):
	""" Evaluate a provided statistic for each observation 
	in the sample

	:param model: SeldonianModel instance
	:param statistic_name: The name of the statistic to evaluate
	:type statistic_name: str, e.g. 'FPR'
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param data_dict: Contains the features and labels 
	:type data_dict: dict

	:return: The evaluated statistic for each observation in the sample
	:rtype: numpy ndarray(float)
	"""

	""" Regression statistics """
	if statistic_name == 'Mean_Squared_Error':
		return vector_Squared_Error(
			model,theta,data_dict['features'],data_dict['labels'])

	if statistic_name == 'Mean_Error':
		return vector_Error(
			model,theta,data_dict['features'],data_dict['labels'])

	""" Classification statistics """
	if statistic_name == 'PR':
		return vector_Positive_Rate(
			model,theta,data_dict['features'])

	if statistic_name == 'NR':
		return vector_Negative_Rate(
			model,theta,data_dict['features'])

	if statistic_name == 'FPR':
		return vector_False_Positive_Rate(
			model,theta,data_dict['features'],data_dict['labels'])

	if statistic_name == 'FNR':
		return vector_False_Negative_Rate(
			model,theta,data_dict['features'],data_dict['labels'])

	if statistic_name == 'TPR':
		return vector_True_Positive_Rate(
			model,theta,data_dict['features'],data_dict['labels'])

	if statistic_name == 'TNR':
		return vector_True_Negative_Rate(
			model,theta,data_dict['features'],data_dict['labels'])

	if statistic_name == 'logistic_loss':
		return vector_logistic_loss(
			model,theta,data_dict['features'],data_dict['labels'])

	""" RL statistics """
	if statistic_name == 'J_pi_new':
		return vector_IS_estimate(
			model,theta,data_dict)

	raise NotImplementedError(
		f"Statistic: {statistic_name} is not implemented")

def evaluate_statistic(model,
	statistic_name,theta,data_dict):
	""" Evaluate a provided statistic for the whole sample provided

	:param model: SeldonianModel instance
	:param statistic_name: The name of the statistic to evaluate
	:type statistic_name: str, e.g. 'FPR' for false positive rate
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param data_dict: Contains the features and labels 
	:type data_dict: dict

	:return: The evaluated statistic over the whole sample
	:rtype: float
	"""
	""" Regression statistics """
	if statistic_name == 'Mean_Squared_Error':
		return Mean_Squared_Error(
			model,theta,data_dict['features'],data_dict['labels'])

	if statistic_name == 'Mean_Error':
		return Mean_Error(
			model,theta,data_dict['features'],data_dict['labels'])

	""" Classification statistics """
	if statistic_name == 'PR':
		return Positive_Rate(
			model,theta,data_dict['features'])

	if statistic_name == 'NR':
		return Negative_Rate(
			model,theta,data_dict['features'])

	if statistic_name == 'FPR':
		return False_Positive_Rate(
			model,theta,data_dict['features'],data_dict['labels'])

	if statistic_name == 'FNR':
		return False_Negative_Rate(
			model,theta,data_dict['features'],data_dict['labels'])

	if statistic_name == 'TPR':
		return True_Positive_Rate(
			model,theta,data_dict['features'],data_dict['labels'])

	if statistic_name == 'TNR':
		return True_Negative_Rate(
			model,theta,data_dict['features'],data_dict['labels'])

	if statistic_name == 'logistic_loss':
		return logistic_loss(
			model,theta,data_dict['features'],data_dict['labels'])

	""" RL statistics """
	if statistic_name == 'J_pi_new':
		return IS_estimate(
			model,theta,data_dict)

	raise NotImplementedError(
		f"Statistic: {statistic_name} is not implemented")

""" Regression """

def Mean_Squared_Error(model,theta,X,Y):
	"""
	Calculate mean squared error over the whole sample

	:param model: SeldonianModel instance
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

def gradient_Mean_Squared_Error(model,theta,X,Y):
	""" Gradient of the mean squared error 

	:param model: SeldonianModel instance
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
	err = prediction-Y
	return 2/n*np.dot(err,X)

def Mean_Error(model,theta,X,Y):
	"""
	Calculate mean error (y_hat-y) over the whole sample

	:param model: SeldonianModel instance
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
	res = sum(prediction-Y)/n
	return res

def vector_Squared_Error(model,theta,X,Y):
	""" Calculate squared error for each observation
	in the dataset
	
	:param model: SeldonianModel instance
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
	
def vector_Error(model,theta,X,Y):
	""" Calculate mean error for each observation
	in the dataset
	
	:param model: SeldonianModel instance
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

def gradient_Bounded_Squared_Error(model,theta,X,Y):
	""" Analytical gradient of the bounded squared error
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarray
	:param Y: The labels
	:type Y: numpy ndarray

	:return: the gradient evaluated at this theta
	:rtype: float
	"""
	n = len(X)
	y_min,y_max = -3,3
	# Want range of Y_hat to be twice that of Y
	# and want size of interval on either side of Y_min and Y_max
	# to be the same. The unique solution to this is:
	s=1.5
	y_hat_min = y_min*(1+s)/2 + y_max*(1-s)/2
	y_hat_max = y_max*(1+s)/2 + y_min*(1-s)/2

	c1 = y_hat_max - y_hat_min
	c2 = -y_hat_min

	Y_hat = model.predict(theta,X) # vector of values
	Y_hat_old = (Y_hat-y_hat_min)/(y_hat_max-y_hat_min)
	sig = model._sigmoid(Y_hat_old)

	term1=Y - (c1*sig-c2)
	term2=-c1*sig*(1-sig)*X[:,0]
	s = sum(term1*term2)
	return -2/n*s


""" Classification """

def logistic_loss(model,theta,X,Y):
	""" Calculate logistic loss 
	on whole sample
	
	:param model: SeldonianModel instance
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

def vector_logistic_loss(model,theta,X,Y):
	""" Calculate logistic loss 
	on each observation in sample
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarra
	:param Y: The labels
	:type Y: numpy ndarray

	:return: array of logistic losses 
	:rtype: numpy ndarray(float)
	"""
	h = model.predict(theta,X)
	res = -Y*np.log(h) - (1.0-Y)*np.log(1.0-h)
	return res		

def gradient_logistic_loss(model,theta,X,Y):
	""" Gradient of logistic loss w.r.t. theta
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarray
	:param Y: The labels
	:type Y: numpy ndarray

	:return: perceptron loss
	:rtype: float
	"""
	
	h = model.predict(theta,X)
	X_withintercept = np.hstack([np.ones((len(X),1)),np.array(X)])
	res = (1/len(X))*np.dot(X_withintercept.T, (h - Y))
	return res

def Positive_Rate(model,theta,X):
	"""
	Calculate positive rate
	for the whole sample.
	This is the sum of probability of each 
	sample being in the positive class
	normalized to the number of predictions 
		
	:param model: SeldonianModel instance:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarray

	:return: Positive rate for whole sample
	:rtype: float between 0 and 1
	"""	
	prediction = model.predict(theta,X)
	return np.sum(prediction)/len(X) # if all 1s then PR=1. 

def Negative_Rate(model,theta,X):
	"""
	Calculate negative rate
	for the whole sample.
	This is the sum of the probability of each 
	sample being in the negative class, which is
	1.0 - probability of being in positive class
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarray

	:return: Negative rate for whole sample
	:rtype: float between 0 and 1
	"""
	prediction = model.predict(theta,X)
	return np.sum(1.0-prediction)/len(X) # if all 1s then PR=1. 

def False_Positive_Rate(model,theta,X,Y):
	"""
	Calculate false positive rate
	for the whole sample.
	
	The is the sum of the probability of each 
	sample being in the positive class when in fact it was in 
	the negative class.
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarray

	:return: False positive rate for whole sample
	:rtype: float between 0 and 1
	"""
	prediction = model.predict(theta,X)
	# Sum the probability of being in positive class
	# subject to the truth being the other class
	neg_mask = Y!=1.0 # this includes false positives and true negatives
	return np.sum(prediction[neg_mask])/len(X[neg_mask])

def False_Negative_Rate(model,theta,X,Y):
	"""
	Calculate false negative rate
	for the whole sample.
	
	The is the sum of the probability of each 
	sample being in the negative class when in fact it was in 
	the positive class.
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarray

	:return: False negative rate for whole sample
	:rtype: float between 0 and 1
	"""
	prediction = model.predict(theta,X)
	# Sum the probability of being in negative class
	# subject to the truth being the positive class
	pos_mask = Y==1.0 # this includes false positives and true negatives
	return np.sum(1.0-prediction[pos_mask])/len(X[pos_mask])

def True_Positive_Rate(model,theta,X,Y):
	"""
	Calculate true positive rate
	for the whole sample.
	
	The is the sum of the probability of each 
	sample being in the positive class when in fact it was in 
	the positive class.
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarray

	:return: False positive rate for whole sample
	:rtype: float between 0 and 1
	"""
	prediction = model.predict(theta,X)
	# Sum the probability of being in positive class
	# subject to the truth being the other class
	pos_mask = Y==1.0 # this includes false positives and true negatives
	return np.sum(prediction[pos_mask])/len(X[pos_mask])

def True_Negative_Rate(model,theta,X,Y):
	"""
	Calculate true negative rate
	for the whole sample.
	
	The is the sum of the probability of each 
	sample being in the negative class when in fact it was in 
	the negative class.
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarray

	:return: False positive rate for whole sample
	:rtype: float between 0 and 1
	"""
	prediction = model.predict(theta,X)
	# Sum the probability of being in negative class
	# subject to the truth being the negative class
	neg_mask = Y!=1.0 # this includes false positives and true negatives
	return np.sum(1.0-prediction[neg_mask])/len(X[neg_mask])
	
def vector_Positive_Rate(model,theta,X):
	"""
	Calculate positive rate
	for each observation.
	
	This is the probability of being positive
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarray

	:return: Positive rate for each observation
	:rtype: numpy ndarray(float between 0 and 1)
	"""
	prediction = model.predict(theta,X) # probability of class 1 for each observation
	return prediction 

def vector_Negative_Rate(model,theta,X):
	"""
	Calculate negative rate
	for each observation.

	This is the probability of being negative
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarray

	:return: Positive rate for each observation
	:rtype: numpy ndarray(float between 0 and 1)
	"""
	prediction = model.predict(theta,X)

	return 1.0 - prediction

def vector_False_Positive_Rate(model,theta,X,Y):
	"""
	Calculate false positive rate
	for each observation

	This is the probability of predicting positive
	subject to the label actually being negative
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarray

	:return: False positive rate for each observation
	:rtype: numpy ndarray(float between 0 and 1)
	"""
	prediction = model.predict(theta,X)
	# The probability of being in positive class
	# subject to the truth being the other class
	neg_mask = Y!=1.0 # this includes false positives and true negatives
	return prediction[neg_mask]

def vector_False_Negative_Rate(model,theta,X,Y):
	"""
	Calculate false negative rate
	for each observation
	
	This is the probability of predicting negative
	subject to the label actually being positive
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarray

	:return: False negative rate for each observation
	:rtype: numpy ndarray(float between 0 and 1)
	"""

	prediction = model.predict(theta,X)
	# The probability of being in positive class
	# subject to the truth being the other class
	pos_mask = Y==1.0 # this includes false positives and true negatives
	return 1.0-prediction[pos_mask]

def vector_True_Positive_Rate(model,theta,X,Y):
	"""
	This is the probability of predicting positive
	subject to the label actually being positive
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarray

	:return: True positive rate for each observation
	:rtype: numpy ndarray(float between 0 and 1)
	"""
	prediction = model.predict(theta,X)
	pos_mask = Y==1.0 # this includes false positives and true negatives
	return prediction[pos_mask]

def vector_True_Negative_Rate(model,theta,X,Y):
	"""
	This is the probability of predicting negative
	subject to the label actually being negative
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarray

	:return: True negative rate for each observation
	:rtype: numpy ndarray(float between 0 and 1)
	"""
	prediction = model.predict(theta,X)
	pos_mask = Y!=1.0 # this includes false positives and true negatives
	return 1.0 - prediction[pos_mask]

""" RL """
def IS_estimate(model,theta,data_dict):
	""" Calculate the unweighted importance sampling estimate
	on all episodes in the dataframe
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param data_dict: The object containing data and metadata

	:return: The IS estimate calculated over all episodes
	:rtype: float
	"""
	episodes = data_dict['episodes']
	if 'gamma' in model.env_kwargs:
		gamma = model.env_kwargs['gamma']
	else:
		gamma = 1.0

	IS_estimate = 0
	for ii, ep in enumerate(episodes):
		pi_news = model.get_probs_from_observations_and_actions(
			theta, ep.observations, ep.actions)
		pi_ratios = pi_news / ep.action_probs
		pi_ratio_prod = np.prod(pi_ratios)
		weighted_return = weighted_sum_gamma(ep.rewards, gamma=gamma)
		IS_estimate += pi_ratio_prod * weighted_return

	IS_estimate /= len(episodes)

	return IS_estimate

def vector_IS_estimate(model, theta, data_dict):
	""" Calculate the unweighted importance sampling estimate
	on each episodes in the dataframe
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param data_dict: The object containing data and metadata

	:return: A vector of IS estimates calculated for each episode
	:rtype: numpy ndarray(float)
	"""
	episodes = data_dict['episodes']
	episodes = data_dict['episodes']
	if 'gamma' in model.env_kwargs:
		gamma = model.env_kwargs['gamma']
	else:
		gamma = 1.0
	result = []
	for ii, ep in enumerate(episodes):
		pi_news = model.get_probs_from_observations_and_actions(
			theta, ep.observations, ep.actions)
		pi_ratio_prod = np.prod(pi_news / ep.action_probs)
		weighted_return = weighted_sum_gamma(ep.rewards, gamma=gamma)
		result.append(pi_ratio_prod * weighted_return)

	return np.array(result)