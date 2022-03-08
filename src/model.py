import numpy as np
from sklearn.linear_model import LinearRegression

class ClassificationModel(object):
	def __init__(self):
		pass

class RegressionModel(object):
	def __init__(self):
		pass

class LRModel(RegressionModel):
	def __init__(self):
		super().__init__()
		self.model_class = LinearRegression

	def predict(self,theta,x):
		""" Given a set of weights, theta,
		and an array of feature vectors, x, which include offsets
		in the first column,
		make prediction using the model """
		return np.dot(theta.T,x.T)

	def sample_Mean_Squared_Error(self,theta,X,Y):
		"""
		Calculate sample mean squared error given a solution and 
		data. Put under regressionmodel
		"""
		n = len(X)
		prediction = self.predict(theta,X) # vector of values
		res = sum(pow(prediction-Y,2))/n
		return res

	def sample_Mean_Error(self,theta,X,Y):
		"""
		Calculate sample mean error given a solution and 
		data. Put under regressionmodel
		"""
		n = len(X)
		prediction = self.predict(theta,X) # vector of values
		res = sum(prediction-Y)/n
		return res

	def sample_from_statistic(self,statistic_name,theta,data_dict):
		
		if statistic_name == 'Mean_Squared_Error':
			return self.vector_Mean_Squared_Error(
				theta,data_dict['features'],data_dict['labels'])

		if statistic_name == 'Mean_Error':
			return self.vector_Mean_Error(
				theta,data_dict['features'],data_dict['labels'])

	def vector_Mean_Squared_Error(self,theta,X,Y):
		""" Calculate sample mean squared error at each 
		point in the dataset and return as a 1-vector
		"""  
		prediction = self.predict(theta, X)
		return pow(prediction-Y,2)
		
	def vector_Mean_Error(self,theta,X,Y):
		""" Calculate sample mean error at each 
		point in the dataset and return as a 1-vector
		"""  
		prediction = self.predict(theta, X)
		return prediction-Y

	def fit(self,X,Y):
		reg = self.model_class().fit(X, Y)
		return np.hstack([np.array(reg.intercept_),reg.coef_[1:]])