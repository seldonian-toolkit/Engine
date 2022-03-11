import numpy as np
from sklearn.linear_model import (LinearRegression,
	LogisticRegression, SGDClassifier)

class SeldonianModel(object):
	def __init__(self):
		pass

class SupervisedModel(SeldonianModel):
	def __init__(self):
		pass

	def fit(self,model,X,Y):
		reg = model.model_class().fit(X, Y)
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
	    n = len(X)
	    Y_mask = Y==1
	    prediction = model.predict(theta,X)
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
		and label = -1.
		Outputs a value between 0 and 1
		"""
		prediction = self.predict(theta,X)
		X_FP = X.loc[(prediction-Y)==2]
		return len(X_FP)/len(X)

	def sample_False_Negative_Rate(self,model,theta,X,Y):
		"""
		Calculate false negative rate
		for the whole sample.
		This happens when prediction = -1
		and label = 1.
		Outputs a value between 0 and 1
		"""

		prediction = self.predict(theta,X)
		X_FN = X.loc[(prediction-Y)==-2]
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
		FP_mask = prediction-Y==2
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
		FN_mask = prediction-Y==-2
		return 1.0*X_FN_mask


class LinearClassifierModel(ClassificationModel):
	def __init__(self):
		super().__init__()
		self.model_class = LinearRegression

	def predict(self,theta,X):
		""" Given a set of weights, theta,
		and an array of feature vectors, X, which include offsets
		in the first column,
		make prediction using the model """

		return np.sign(np.dot(theta.T,X.T)) # -1 or 1


class LogisticRegressionModel(ClassificationModel):
	def __init__(self):
		super().__init__()
		self.model_class = LogisticRegression

	def predict(self,theta,X):
		""" Given a set of weights, theta,
		and an array of feature vectors, X, which include offsets
		in the first column,
		make prediction using the model """

		return 1/(1+np.exp(-1*np.dot(theta.T,X.T)))


