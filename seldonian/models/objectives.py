""" Objective functions """

import autograd.numpy as np   # Thinly-wrapped version of Numpy
import math

from seldonian.utils.stats_utils import weighted_sum_gamma


def batcher(func,N,batch_size,num_batches):
	""" Calls function num_batches times,
	batching up the inputs to the objective function
	or measure function.

	:param func: The function you want to call
	:param N: The total number of datapoints
	:type N: int
	:param batch_size: The size of each batch
	:type batch_size: int
	:param num_batches: The number of batches
	:type num_batches: int

	:return: The value from calling function on
		the batched up data 
	"""
	def wrapper(*args, **kw):   
		regime = kw['regime']
		model=args[0]
		theta=args[1]
		if regime == 'supervised_learning':
			features=args[2]
			labels=args[3]
		elif regime == 'reinforcement_learning':
			episodes = args[2]
		print(f"Have {num_batches} batches of size {batch_size} in safety test")
		if num_batches > 1:
			res = np.zeros(N)
			batch_start = 0 
			for i in range(num_batches):
				batch_end = batch_start + batch_size
				if regime == 'supervised_learning':
					if type(features) == list:
						features_batch = [x[batch_start:batch_end] for x in features]
					else:
						features_batch = features[batch_start:batch_end]

					labels_batch = labels[batch_start:batch_end]
					batch_args = [model,theta,features_batch,labels_batch]
					
				elif regime == 'reinforcement_learning':
					episodes_batch = episodes[batch_start:batch_end]
					batch_args = [model,theta,episodes_batch]
				
				res[batch_start:batch_end] = func(
						*batch_args,
						**kw)

				batch_start=batch_end
		else:

			res = func(*args,**kw)  
		return res
	return wrapper

def sample_from_statistic(model,
	statistic_name,theta,data_dict,**kwargs):
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
	branch = kwargs['branch']
	dataset = kwargs['dataset']
	regime = dataset.regime
	msr_func_kwargs = {'regime':regime}

	if regime == 'supervised_learning':
		args = [model,theta,data_dict['features'],data_dict['labels']]
		sub_regime = dataset.meta_information['sub_regime']
		msr_func_kwargs['sub_regime'] = sub_regime
		if 'class_index' in kwargs:
			msr_func_kwargs['class_index'] = kwargs['class_index']
		if 'cm_true_index' in kwargs:
			msr_func_kwargs['l_i'] = kwargs['cm_true_index']
		if 'cm_pred_index' in kwargs:
			msr_func_kwargs['l_k'] = kwargs['cm_pred_index']
	
	elif regime == 'reinforcement_learning':
		episodes = data_dict['episodes']
		weighted_returns = data_dict['weighted_returns']
		args = [model,theta,episodes,weighted_returns]

	msr_func = measure_function_vector_mapper[statistic_name]

	if branch == 'candidate_selection':
		return msr_func(*args,**msr_func_kwargs)

	elif branch == 'safety_test':
		num_datapoints = dataset.num_datapoints
		if 'batch_size_safety' in kwargs:
			if kwargs['batch_size_safety'] is None:
				batch_size_safety = num_datapoints
				num_batches = 1
			else:	
				batch_size_safety = kwargs['batch_size_safety'] 
				num_batches = math.ceil(num_datapoints / batch_size_safety)

		else:
			batch_size_safety = num_datapoints
			num_batches = 1

		return batcher(
			msr_func,
			N=num_datapoints,
			batch_size=batch_size_safety,
			num_batches=num_batches)(*args,**msr_func_kwargs)

def evaluate_statistic(model,
	statistic_name,theta,data_dict,**kwargs):
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
	branch = kwargs['branch']
	dataset = kwargs['dataset']
	regime = dataset.regime
	msr_func_kwargs = {'regime':regime}

	if regime == 'supervised_learning':
		args = [model,theta,data_dict['features'],data_dict['labels']]
		sub_regime = dataset.meta_information['sub_regime']
		msr_func_kwargs['sub_regime'] = sub_regime
		if 'class_index' in kwargs:
			msr_func_kwargs['class_index'] = kwargs['class_index']
		if 'cm_true_index' in kwargs:
			msr_func_kwargs['l_i'] = kwargs['cm_true_index']
		if 'cm_pred_index' in kwargs:
			msr_func_kwargs['l_k'] = kwargs['cm_pred_index']
	
	elif regime == 'reinforcement_learning':
		episodes = data_dict['episodes']
		weighted_returns = data_dict['weighted_returns']
		args = [model,theta,episodes,weighted_returns]

	msr_func = measure_function_mapper[statistic_name]

	if branch == 'candidate_selection':
		return msr_func(*args,**msr_func_kwargs)

	elif branch == 'safety_test':
		num_datapoints = dataset.num_datapoints
		if 'batch_size_safety' in kwargs:
			if kwargs['batch_size_safety'] is None:
				batch_size_safety = num_datapoints
				num_batches = 1
			else:	
				batch_size_safety = kwargs['batch_size_safety'] 
				num_batches = math.ceil(num_datapoints / batch_size_safety)

		else:
			batch_size_safety = num_datapoints
			num_batches = 1

		return batcher(
			msr_func,
			N=num_datapoints,
			batch_size=batch_size_safety,
			num_batches=num_batches)(*args,**msr_func_kwargs)

""" Regression """

def Mean_Squared_Error(model,theta,X,Y,**kwargs):
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
	n = len(Y) # Y guaranteed to be a numpy array, X isn't.
	prediction = model.predict(theta,X) # vector of values
	res = sum(pow(prediction-Y,2))/n

	return res

def gradient_Mean_Squared_Error(model,theta,X,Y,**kwargs):
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
	if type(X) == list:
		raise NotImplementedError(
			"This function is not supported when features are in a list. "
			"Convert features to a numpy array if possible or use autodiff "
			" to get the gradient.")
	n = len(Y)
	prediction = model.predict(theta,X) # vector of values
	err = prediction-Y
	X_withintercept = np.hstack([np.ones((n,1)),np.array(X)])
	return 2/n*np.dot(err,X_withintercept)

def Mean_Error(model,theta,X,Y,**kwargs):
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

def vector_Squared_Error(model,theta,X,Y,**kwargs):
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
	
def vector_Error(model,theta,X,Y,**kwargs):
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

def gradient_Bounded_Squared_Error(model,theta,X,Y,**kwargs):
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

def binary_logistic_loss(model,theta,X,Y,**kwargs):
	""" Calculate average logistic loss 
	over all data points for binary classification
	
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
	Y_pred = model.predict(theta,X)
	# binary 
	res = np.mean(-Y*np.log(Y_pred) - (1.0-Y)*np.log(1.0-Y_pred))
	return res

def gradient_binary_logistic_loss(model,theta,X,Y,**kwargs):
	""" Gradient of binary logistic loss w.r.t. theta
	
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

def multiclass_logistic_loss(model,theta,X,Y,**kwargs):
	""" Calculate average logistic loss 
	over all data points for multi-class classification
	
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
	# Negative log likelihood 
	# In the multi-class setting, y_pred is an i x k matrix 
	# where i is the number of samples and k is the number of classes
	# Each entry is the probability of predicting the kth class 
	# for the ith sample. We need to get the probability of predicting
	# the true class for each sample and then take the sum of the 
	# logs of that.
	Y_pred = model.predict(theta,X)	
	N = len(Y) 
	probs_trueclasses = Y_pred[np.arange(N),Y.astype('int')]
	return -1/N*sum(np.log(probs_trueclasses))
		
def Positive_Rate(model,theta,X,Y,**kwargs):
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
	if 'class_index' in kwargs:
		return _Positive_Rate_multiclass(model,theta,X,Y,
			class_index=kwargs['class_index'])
	else:
		return _Positive_Rate_binary(model,theta,X,Y)

def _Positive_Rate_binary(model,theta,X,Y,**kwargs):
	prediction = model.predict(theta,X)
	return np.sum(prediction)/len(X) # if all 1s then PR=1.

def _Positive_Rate_multiclass(model,theta,X,Y,class_index,**kwargs):
	prediction = model.predict(theta,X)
	return np.sum(prediction[:,class_index])/len(X) # if all 1s then PR=1. 

def Negative_Rate(model,theta,X,Y,**kwargs):
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
	if 'class_index' in kwargs:
		return _Negative_Rate_multiclass(model,theta,X,Y,
			class_index=kwargs['class_index'])
	else:
		return _Negative_Rate_binary(model,theta,X,Y)

def _Negative_Rate_binary(model,theta,X,Y,**kwargs):
	# Average probability of predicting the negative class
	prediction = model.predict(theta,X)
	return np.sum(1.0-prediction)/len(X) # if all 1s then PR=1. 

def _Negative_Rate_multiclass(model,theta,X,Y,class_index,**kwargs):
	# Average probability of predicting class!=class_index
	prediction = model.predict(theta,X)
	return np.sum(1.0-prediction[:,class_index])/len(X)  

def False_Positive_Rate(model,theta,X,Y,**kwargs):
	"""
	Calculate probabilistic average false positive rate
	over the whole sample. The is the average probability of 
	predicting the positive class when the true label was
	the negative class.
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarray

	:return: Average false positive rate 
	:rtype: float between 0 and 1
	"""
	if 'class_index' in kwargs:
		return _False_Positive_Rate_multiclass(model,theta,X,Y,
			class_index=kwargs['class_index'])
	else:
		return _False_Positive_Rate_binary(model,theta,X,Y)

def _False_Positive_Rate_binary(model,theta,X,Y,**kwargs):
	# Average probability of predicting positive class
	# subject to the truth being the other class
	prediction = model.predict(theta,X)
	neg_mask = Y!=1.0 
	return np.sum(prediction[neg_mask])/len(X[neg_mask])

def _False_Positive_Rate_multiclass(model,theta,X,Y,class_index,**kwargs):
	# Sum the probability of predicting class=class_index 
	# subject to the true label being any other class
	prediction = model.predict(theta,X)

	neg_mask = Y!=class_index 
	return np.sum(prediction[:,class_index][neg_mask])/len(X[neg_mask])

def False_Negative_Rate(model,theta,X,Y,**kwargs):
	"""
	Calculate probabilistic average false negative rate
	over the whole sample. The is the average probability  
	of predicting the negative class when truth was positive class.
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarray

	:return: Average false negative rate 
	:rtype: float between 0 and 1
	"""
	if 'class_index' in kwargs:
		return _False_Negative_Rate_multiclass(model,theta,X,Y,
			class_index=kwargs['class_index'])
	else:
		return _False_Negative_Rate_binary(model,theta,X,Y)

def _False_Negative_Rate_binary(model,theta,X,Y,**kwargs):
	# Average probability of being in negative class
	# subject to the truth being the positive class
	prediction = model.predict(theta,X)
	pos_mask = Y==1.0 
	return np.sum(1.0-prediction[pos_mask])/len(X[pos_mask])

def _False_Negative_Rate_multiclass(model,theta,X,Y,class_index,**kwargs):
	# Average probability of not having class=class_index 
	# subject to the truth being class=class_index
	prediction = model.predict(theta,X)
	pos_mask = Y==class_index
	return np.sum(1.0-prediction[:,class_index][pos_mask])/len(X[pos_mask])

def True_Positive_Rate(model,theta,X,Y,**kwargs):
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
	if 'class_index' in kwargs:
		return _True_Positive_Rate_multiclass(model,theta,X,Y,
			class_index=kwargs['class_index'])
	else:
		return _True_Positive_Rate_binary(model,theta,X,Y)

def _True_Positive_Rate_binary(model,theta,X,Y,**kwargs):
	# Average probability of predicting the positive class
	# subject to the true label being the positive class
	prediction = model.predict(theta,X)
	pos_mask = Y==1.0 
	return np.sum(prediction[pos_mask])/len(X[pos_mask])

def _True_Positive_Rate_multiclass(model,theta,X,Y,class_index,**kwargs):
	# Average probability of predicting class=class_index
	# subject to the true label having class=class_index
	prediction = model.predict(theta,X)
	pos_mask = Y==class_index
	return np.sum(prediction[:,class_index][pos_mask])/len(X[pos_mask])

def True_Negative_Rate(model,theta,X,Y,**kwargs):
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
	if 'class_index' in kwargs:
		return _True_Negative_Rate_multiclass(model,theta,X,Y,
			class_index=kwargs['class_index'])
	else:
		return _True_Negative_Rate_binary(model,theta,X,Y)
	
def _True_Negative_Rate_binary(model,theta,X,Y,**kwargs):
	# Average probability of being in negative class
	# subject to the truth being the negative class
	prediction = model.predict(theta,X)
	neg_mask = Y!=1.0 
	return np.sum(1.0-prediction[neg_mask])/len(X[neg_mask])

def _True_Negative_Rate_multiclass(model,theta,X,Y,class_index,**kwargs):
	# Average probability of class!=class_index
	# subject to the truth being class!=class_index
	prediction = model.predict(theta,X)
	neg_mask = Y!=class_index
	return np.sum(1.0-prediction[:,class_index][neg_mask])/len(X[neg_mask])

def True_Positive_Rate(model,theta,X,Y,**kwargs):
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
	if 'class_index' in kwargs:
		return _True_Positive_Rate_multiclass(model,theta,X,Y,
			class_index=kwargs['class_index'])
	else:
		return _True_Positive_Rate_binary(model,theta,X,Y)

def Accuracy(model,theta,X,Y,**kwargs):
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
	if kwargs['sub_regime'] == 'multiclass_classification':
		return _Accuracy_multiclass(model,theta,X,Y,**kwargs)
	else:
		return _Accuracy_binary(model,theta,X,Y,**kwargs)
	
def _Accuracy_binary(model,theta,X,Y,**kwargs):
	""" Calculate accuracy
	over all data points for binary classification
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarray
	:param Y: The labels
	:type Y: numpy ndarray

	:return: accuracy
	:rtype: float
	"""
	n = len(X)
	Y_pred_probs = model.predict(theta,X)
	v = np.where(Y!=1,1.0-Y_pred_probs,Y_pred_probs)
	return np.sum(v)/n

def _Accuracy_multiclass(model,theta,X,Y,**kwargs):
	""" Calculate accuracy
	over all data points for multi-class classification
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarray
	:param Y: The labels
	:type Y: numpy ndarray

	:return: accuracy
	:rtype: float
	"""
	n = len(X)
	Y_pred_probs = model.predict(theta,X)
	return np.sum(Y_pred_probs[np.arange(n),Y])/n

def confusion_matrix(model,theta,X,Y,l_i,l_k,**kwargs):
	"""
	Get the probability of predicting class label l_k 
	if the true class label was l_i. This is the C[l_i,l_k]
	element of the confusion matrix, C. Let:
		i = number of datapoints
		j = number of features (including bias term, if provied)
		k = number of classes
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: array of shape (j,k)
	:param X: The features
	:type X: array of shape (i,j)
	:param Y: The labels
	:type Y: array of shape (i,k)
	:param l_i: The index in the confusion matrix
		corresponding to the true label (row)
	:type l_i: int
	:param l_k: The index in the confusion matrix
		corresponding to the predicted label (column)
	:type l_k: int

	:return: The element 
	:rtype: float
	"""
	Y_pred = model.predict(theta,X) # i x k
	true_mask = Y == l_i # length i
	N_mask = sum(true_mask)

	res = sum(Y_pred[:,l_k][true_mask])/N_mask 
	return res
	
def vector_Positive_Rate(model,theta,X,Y,**kwargs):
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
	if 'class_index' in kwargs:
		return _vector_Positive_Rate_multiclass(
			model,theta,X,Y,class_index=kwargs['class_index'])
	else:
		return _vector_Positive_Rate_binary(
			model,theta,X,Y)

def _vector_Positive_Rate_binary(model,theta,X,Y,**kwargs):
	# probability of class 1 for each observation
	prediction = model.predict(theta,X) 
	return prediction 

def _vector_Positive_Rate_multiclass(model,theta,X,Y,class_index,**kwargs):
	# probability of class==class_index for each observation
	prediction = model.predict(theta,X) 
	return prediction[:,class_index]

def vector_Negative_Rate(model,theta,X,Y,**kwargs):
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
	if 'class_index' in kwargs:
		return _vector_Negative_Rate_multiclass(
			model,theta,X,Y,class_index=kwargs['class_index'])
	else:
		return _vector_Negative_Rate_binary(
			model,theta,X,Y)

def _vector_Negative_Rate_binary(model,theta,X,Y,**kwargs):
	# probability of class 0 for each observation
	prediction = model.predict(theta,X)
	return 1.0 - prediction

def _vector_Negative_Rate_multiclass(model,theta,X,Y,class_index,**kwargs):
	# probability of class!=class_index for each observation
	prediction = model.predict(theta,X)
	return 1.0 - prediction[:,class_index]

def vector_False_Positive_Rate(model,theta,X,Y,**kwargs):
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
	if 'class_index' in kwargs:
		return _vector_False_Positive_Rate_multiclass(
			model,theta,X,Y,class_index=kwargs['class_index'])
	else:
		return _vector_False_Positive_Rate_binary(
			model,theta,X,Y)

def _vector_False_Positive_Rate_binary(model,theta,X,Y,**kwargs):
	# The probability the model predicts being in this class
	# subject to the truth being in any other class
	prediction = model.predict(theta,X)
	neg_mask = Y!=1.0 # this includes false positives and true negatives
	return prediction[neg_mask]

def _vector_False_Positive_Rate_multiclass(model,theta,X,Y,class_index,**kwargs):
	# The probability the model predicts being in this class
	# subject to the truth being in any other class
	prediction = model.predict(theta,X)
	other_mask = Y != class_index
	return prediction[:,class_index][other_mask]

def vector_False_Negative_Rate(model,theta,X,Y,**kwargs):
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
	if 'class_index' in kwargs:
		return _vector_False_Negative_Rate_multiclass(
			model,theta,X,Y,class_index=kwargs['class_index'])
	else:
		return _vector_False_Negative_Rate_binary(
			model,theta,X,Y)

def _vector_False_Negative_Rate_binary(model,theta,X,Y,**kwargs):
	# The probability of being in positive class
	# subject to the truth being the other class
	prediction = model.predict(theta,X)
	pos_mask = Y==1.0 # this includes false positives and true negatives
	return 1.0-prediction[pos_mask]

def _vector_False_Negative_Rate_multiclass(model,theta,X,Y,class_index,**kwargs):
	# The probability the model predicts not being in this class
	# subject to the truth being in this class
	prediction = model.predict(theta,X)
	pos_mask = Y==class_index # this includes false positives and true negatives
	return (1.0-prediction[:,class_index])[pos_mask]

def vector_True_Positive_Rate(model,theta,X,Y,**kwargs):
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
	if 'class_index' in kwargs:
		return _vector_True_Positive_Rate_multiclass(
			model,theta,X,Y,class_index=kwargs['class_index'])
	else:
		return _vector_True_Positive_Rate_binary(
			model,theta,X,Y)
	
def _vector_True_Positive_Rate_binary(model,theta,X,Y,**kwargs):
	"""
	This is the probability of predicting positive
	subject to the label actually being positive
	"""
	prediction = model.predict(theta,X)
	pos_mask = Y==1.0 # this includes false positives and true negatives
	return prediction[pos_mask]

def _vector_True_Positive_Rate_multiclass(model,theta,X,Y,class_index,**kwargs):
	"""
	This is the probability of predicting this class
	subject to the label actually being this class
	"""
	prediction = model.predict(theta,X)
	pos_mask = Y==class_index # this includes false positives and true negatives
	return (prediction[:,class_index])[pos_mask]
	
def vector_True_Negative_Rate(model,theta,X,Y,**kwargs):
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
	if 'class_index' in kwargs:
		return _vector_True_Negative_Rate_multiclass(
			model,theta,X,Y,class_index=kwargs['class_index'])
	else:
		return _vector_True_Negative_Rate_binary(
			model,theta,X,Y)

def _vector_True_Negative_Rate_binary(model,theta,X,Y,**kwargs):
	"""
	This is the probability of predicting negative
	subject to the label actually being negative
	"""
	prediction = model.predict(theta,X)
	neg_mask = Y!=1.0 
	return 1.0 - prediction[neg_mask]

def _vector_True_Negative_Rate_multiclass(model,theta,X,Y,class_index,**kwargs):
	"""
	This is the probability of predicting not this class
	subject to the true label not being this class
	"""
	prediction = model.predict(theta,X)
	neg_mask = Y!=class_index 
	return (1.0 - prediction[:,class_index])[neg_mask]

def vector_Accuracy(model,theta,X,Y,**kwargs):
	"""
	This is the probability of predicting the
	correct label
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarray

	:return: True negative rate for each observation
	:rtype: numpy ndarray(float between 0 and 1)
	"""
	if kwargs['sub_regime'] == 'multiclass_classification':
		return _vector_Accuracy_multiclass(
			model,theta,X,Y,**kwargs)
	else:
		return _vector_Accuracy_binary(
			model,theta,X,Y,**kwargs)

def _vector_Accuracy_binary(model,theta,X,Y,**kwargs):
	""" Calculate vector of probability of 
	predicting the true label
	
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
	Y_pred_probs = model.predict(theta,X)
	# Get probabilities of true positives and true negatives
	# Use the vector Y_pred as it already has the true positive
	# probs. Just need to replace the probabilites in the neg mask with 1-prob
	return np.where(Y!=1,1.0-Y_pred_probs,Y_pred_probs)

def _vector_Accuracy_multiclass(model,theta,X,Y,**kwargs):
	""" Calculate vector of probability of 
	predicting the true label for each data point
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param X: The features
	:type X: numpy ndarray
	:param Y: The labels
	:type Y: numpy ndarray

	:return: accuracy
	:rtype: float
	"""
	n = len(X)
	Y_pred_probs = model.predict(theta,X)
	return Y_pred_probs[np.arange(n),Y]

def vector_confusion_matrix(model,theta,X,Y,l_i,l_k,**kwargs):
	"""
	Get the probability of predicting class label l_k 
	if the true class label was l_i. This is the C[l_i,l_k]
	element of the confusion matrix, C. Let:
		i = number of datapoints
		j = number of features (including bias term, if provied)
		k = number of classes
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: array of shape (j,k)
	:param X: The features
	:type X: array of shape (i,j)
	:param Y: The labels
	:type Y: array of shape (i,k)
	:param l_i: The index in the confusion matrix
		corresponding to the true label (row)
	:type l_i: int
	:param l_k: The index in the confusion matrix
		corresponding to the predicted label (column)
	:type l_k: int

	:return: Array of the C[l_i,l_k] for each observation 
	:rtype: array of floats
	"""
	Y_pred = model.predict(theta,X) # i x k
	true_mask = Y == l_i # length i
	
	N_mask = sum(true_mask)
	res = Y_pred[:,l_k][true_mask]
	return res

""" RL """
def IS_estimate(model,theta,episodes,weighted_returns=None,**kwargs):
	""" Calculate the unweighted importance sampling estimate
	on all episodes in the dataframe
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param episodes: List of episodes

	:return: The IS estimate calculated over all episodes
	:rtype: float
	"""
	# Possible that weighted returns were calculated ahead of time.
	# If not, then calculated them here
	if weighted_returns is None:
		if 'gamma' in model.env_kwargs:
			gamma = model.env_kwargs['gamma']
		else:
			gamma = 1.0
		weighted_returns = [weighted_sum_gamma(ep.rewards, gamma=gamma) for ep in episodes]

	IS_estimate = 0
	for ii,ep in enumerate(episodes):
		pi_news = model.get_probs_from_observations_and_actions(
			theta, ep.observations, ep.actions)
		pi_ratios = pi_news / ep.action_probs
		pi_ratio_prod = np.prod(pi_ratios)
		
		IS_estimate += pi_ratio_prod * weighted_returns[ii]

	IS_estimate /= len(episodes)

	return IS_estimate

def vector_IS_estimate(model,theta,episodes,weighted_returns,**kwargs):
	""" Calculate the unweighted importance sampling estimate
	on each episodes in the dataframe
	
	:param model: SeldonianModel instance
	:param theta: The parameter weights
	:type theta: numpy ndarray
	:param episodes: List of episodes

	:return: A vector of IS estimates calculated for each episode
	:rtype: numpy ndarray(float)
	"""
	
	result = []
	for ii,ep in enumerate(episodes):
		pi_news = model.get_probs_from_observations_and_actions(
			theta, ep.observations, ep.actions)
		pi_ratio_prod = np.prod(pi_news / ep.action_probs)
		result.append(pi_ratio_prod * weighted_returns[ii])

	return np.array(result)

""" Measure function mappers """ 
measure_function_vector_mapper = {
	'Mean_Squared_Error':vector_Squared_Error,
	'Mean_Error':vector_Error,
	'PR':vector_Positive_Rate,
	'NR':vector_Negative_Rate,
	'FPR':vector_False_Positive_Rate,
	'FNR':vector_False_Negative_Rate,
	'TPR':vector_True_Positive_Rate,
	'TNR':vector_True_Negative_Rate,
	'ACC':vector_Accuracy,
	'J_pi_new':vector_IS_estimate,
}

measure_function_mapper = {
	'Mean_Squared_Error':Mean_Squared_Error,
	'Mean_Error':Mean_Error,
	'PR':Positive_Rate,
	'NR':Negative_Rate,
	'FPR':False_Positive_Rate,
	'FNR':False_Negative_Rate,
	'TPR':True_Positive_Rate,
	'TNR':True_Negative_Rate,
	'ACC':Accuracy,
	'J_pi_new':IS_estimate,
}