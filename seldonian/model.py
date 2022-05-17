import autograd.numpy as np   # Thinly-wrapped version of Numpy
from sklearn.linear_model import (LinearRegression,
	LogisticRegression, SGDClassifier)
from seldonian.stats_utils import weighted_sum_gamma
from functools import partial, lru_cache

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
				theta,data_dict['features'].values,data_dict['labels'].values)

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
		return np.dot(X,theta)


class ClassificationModel(SupervisedModel):
	def __init__(self):
		super().__init__()

	def sample_from_statistic(self,
		statistic_name,model,theta,data_dict):
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
		prediction = model.predict(theta,X)
		acc = np.mean(1.0*prediction==Y)
		return acc

	def perceptron_loss(self,model,theta,X,Y):
		prediction = model.predict(theta,X)
		res = np.mean(prediction!=Y) # 0 if all correct, 1 if all incorrect
		return res

	def logistic_loss(self,model,theta,X,Y):
		h = 1/(1+np.exp(-1.0*np.dot(X,theta)))
		res = np.mean(-Y*np.log(h) - (1.0-Y)*np.log(1.0-h))
		return res

	def Vector_logistic_loss(self,model,theta,X,Y):
		h = 1/(1+np.exp(-1.0*np.dot(X,theta)))
		res = -Y*np.log(h) - (1.0-Y)*np.log(1.0-h)
		return res		

	def sample_Positive_Rate(self,model,theta,X,Y):
		"""
		Calculate positive rate
		for the whole sample.
		This happens when prediction = 1
		Outputs a value between 0 and 1
		"""
		prediction = self.predict(theta,X.values)
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

	def Vector_Positive_Rate(self,model,theta,X):
		"""
		Calculate positive rate
		for each observation.
		This happens when prediction = 1
		Outputs a value between 0 and 1
		"""
		prediction = self.predict(theta,X)
		P_mask = prediction==1.0
		res = 1.0*P_mask
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
		prediction = np.sign(np.dot(X,theta)) # -1 or 1
		return prediction

	def default_objective(self,model,theta,X,Y):
		return self.perceptron_loss(model,theta,X,Y)


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

		h = 1/(1+np.exp(-1.0*np.dot(X,theta)))

		prediction = h>=0.5
		return prediction

	def predict_proba(self,theta,X):
		""" Given a model class instance predict the class label 
		given features X"""
		h = 1/(1+np.exp(-1.0*np.dot(X,theta)))

		return h

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
		model.theta = theta
		pi_ratios = list(map(model.apply_policy,
					dataset.df['O'].values,
					dataset.df['A'].values))/dataset.df['pi'].values
		model.theta = None
		model.denom.cache_clear()
		model.arg.cache_clear()
		split_indices_by_episode = np.unique(dataset.df['episode_index'].values,
			return_index=True)[1][1:]
		pi_ratios_by_episode = np.split(pi_ratios, split_indices_by_episode) # this is a list
		products_by_episode = np.array(list(map(np.prod,pi_ratios_by_episode)))
		
		# Weighted rewards
		gamma = self.environment.gamma
		rewards_by_episode = np.split(dataset.df['R'].values,split_indices_by_episode)
		weighted_reward_sums = np.array(list(map(weighted_sum_gamma,
			rewards_by_episode,gamma*np.ones_like(rewards_by_episode))))
		result = sum(products_by_episode*weighted_reward_sums)/len(pi_ratios_by_episode)
		return result 


	def vector_IS_estimate(self,model,theta,data_dict):
		model.theta = theta
		pi_ratios = list(map(model.apply_policy,
					data_dict['dataframe']['O'].values,
					data_dict['dataframe']['A'].values))/data_dict['dataframe']['pi'].values
		model.theta = None
		model.denom.cache_clear()
		model.arg.cache_clear()
		split_indices_by_episode = np.unique(data_dict['dataframe']['episode_index'].values,
			return_index=True)[1][1:]
		pi_ratios_by_episode = np.split(pi_ratios, split_indices_by_episode) # this is a list
		products_by_episode = np.array(list(map(np.prod,pi_ratios_by_episode)))
		result = (
			products_by_episode*data_dict['reward_sums_by_episode'].values
			)
		return result

class TabularSoftmaxModel(RLModel):
	# Call this tabular softmax
	def __init__(self,environment):
		self.environment = environment
		self.theta = None

	@lru_cache
	def denom(self,state):
		return np.sum(np.exp(self.theta[state*4+self.environment.actions]))

	@lru_cache
	def arg(self,state,action):
		return self.theta[state*4+action]

	def apply_policy(self,state,action):
		# Call pi or get action probability
		""" Apply the softmax policy given a state and action
		as well as the set of policy parameters, theta.
		Theta is a flattened parameter vector """
		state = int(state)
		action = int(action)
		
		return np.exp(self.arg(state,action))/self.denom(state)

	def default_objective(self,model,theta,dataset):
		return self.IS_estimate(model,theta,dataset)

class LinearSoftmaxModel(RLModel):
	# Call this linear softmax. 
	# Must provide an objective (or default objective)
	# and an apply policy method
	def __init__(self,environment):
		self.environment = environment
		self.theta = None

	@lru_cache
	def denom(self,state):
		return np.sum(np.exp(self.theta[state*4+self.environment.actions]))

	@lru_cache
	def arg(self,state,action):
		return self.theta[state*4+action]

	def apply_policy(self,state,action):
		print("In apply policy. State, action:")
		print(state,action)
		print(type(state))
		# Call pi or get action probability
		""" Apply the softmax policy given a state and action
		as well as the set of policy parameters """
		logp = self.environment.policy.log_probabilty(state, action)
		return np.exp(logp)


	def default_objective(self,model,theta,dataset):
		return self.IS_estimate(model,theta,dataset)
