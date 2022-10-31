# pytorch_model.py
### A simple single layer Pytorch model implementing linear regression 

import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd.extend import primitive, defvjp
from seldonian.models.models import SupervisedModel

import torch
import torch.nn as nn

@primitive
def pytorch_predict(theta,X,model,**kwargs):
	""" Do a forward pass through the PyTorch model.
	Must convert back to numpy array before returning 

	:param theta: model weights
	:type theta: numpy ndarray

	:param X: model features
	:type X: numpy ndarray

	:param model: An instance of a class inheriting from
		SupervisedPytorchBaseModel 

	:return pred_numpy: model predictions 
	:rtype pred_numpy: numpy ndarray same shape as labels
	"""
	# First update model weights
	
	model.update_model_params(theta,**kwargs)
	# Do the forward pass
	pred = model.forward_pass(X,**kwargs)
	# set the predictions attribute of the model
	model.predictions = pred
	# Convert predictions into a numpy array
	pred_numpy = pred.detach().numpy()
	return pred_numpy

def pytorch_predict_vjp(ans,theta,X,model):
	""" Do a backward pass through the PyTorch model,
	obtaining the Jacobian d pred / dtheta. 
	Must convert back to numpy array before returning 

	:param theta: model weights
	:type theta: numpy ndarray

	:param X: model features
	:type X: numpy ndarray

	:param model: An instance of a class inheriting from
		SupervisedPytorchBaseModel 

	:return fn: A function representing the vector Jacobian operator
	"""
	def fn(v):
		# v is a vector of shape ans, the return value of mypredict()
		# return a 1D array [dF_i/dtheta[0],dF_i/dtheta[1],dF_i/dtheta[2]],
		# where i is the data row index
		model.zero_gradients()
		external_grad = torch.tensor(v)
		dpred_dtheta = model.backward_pass(external_grad)
		return np.array(dpred_dtheta)
	return fn

# Link the predict function with its gradient,
# telling autograd not to look inside either of these functions
defvjp(pytorch_predict,pytorch_predict_vjp)

class SupervisedPytorchBaseModel(SupervisedModel):
	def __init__(self,input_dim,output_dim,**kwargs):
		""" Base class for Supervised learning Seldonian
		models implemented in Pytorch

		:param input_dim: Number of features
		:param output_dim: Size of output layer (number of label columns)
		"""
		super().__init__()
		self.input_dim=input_dim
		self.output_dim=output_dim
		self.pytorch_model = self.create_model(**kwargs)

	def predict(self,theta,X,**kwargs):
		""" Do a forward pass through the PyTorch model.
		Must convert back to numpy array before returning 

		:param theta: model weights
		:type theta: numpy ndarray

		:param X: model features
		:type X: numpy ndarray

		:return pred_numpy: model predictions 
		:rtype pred_numpy: numpy ndarray same shape as labels
		"""
		return pytorch_predict(theta,X,self)

	def create_model(self,**kwargs):
		""" Create the pytorch model and return it
		"""
		raise NotImplementedError("Implement this method in child class")

	def update_model_params(self,theta,**kwargs):
		""" Update weights of PyTorch model using theta,
		the weights from the previous step of gradient descent

		:param theta: model weights
		:type theta: numpy ndarray
		"""
		raise NotImplementedError("Implement this method in child class")

	def zero_gradients(self,*kwargs):
		""" Zero out the gradients of all model parameters """
		raise NotImplementedError("Implement this method in child class")

	def forward_pass(self,X,**kwargs):
		""" Do a forward pass through the PyTorch model and return the 
		model outputs (predicted labels). The outputs should be the same shape 
		as the true labels
	
		:param X: model features
		:type X: numpy ndarray
		"""
		raise NotImplementedError("Implement this method in child class")

	def backward_pass(self,external_grad,**kwargs):
		""" Do a backward pass through the PyTorch model and return the
		(vector) gradient of the model with respect to theta as a numpy ndarray

		:param external_grad: The gradient of the model with respect to itself
			see: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#differentiation-in-autograd
			for more details
		:type external_grad: torch.Tensor 
		"""
		raise NotImplementedError("Implement this method in child class")

class PytorchLRTestModel(SupervisedPytorchBaseModel):
	def __init__(self,input_dim,output_dim):
		""" Implements linear regression using a single Pytorch linear layer

		:param input_dim: Number of features
		:param output_dim: Size of output layer (number of label columns)
		"""
		super().__init__(input_dim,output_dim)
		self.param_sizes = self.get_param_sizes()
		print("param param_sizes:")
		print(self.param_sizes)
		self.has_intercept=True

	def create_model(self,**kwargs):
		""" Create the pytorch model and return it
		"""
		return torch.nn.Linear(self.input_dim, self.output_dim)

	def get_param_sizes(self):
		param_sizes = []
		for param in self.pytorch_model.parameters():
			if param.requires_grad:
				param_sizes.append(param.numel())
		return param_sizes

	def update_model_params(self,theta,**kwargs):
		""" Update all model parameters using theta, which must be reshaped

		:param theta: model weights
		:type theta: numpy ndarray
		"""
		# Update model parameters using flattened array
		with torch.no_grad():
			i = 0
			startindex = 0
			for param in self.pytorch_model.parameters():
				if param.requires_grad:
					nparams = self.param_sizes[i]
					param_shape = param.shape
					theta_numpy = theta[startindex:startindex+nparams]
					theta_torch = torch.from_numpy(theta_numpy).view(param_shape)
					param.copy_(theta_torch)
					i+=1
					startindex+=nparams
		return

	def zero_gradients(self):
		""" Zero out gradients of all model parameters """
		for param in self.pytorch_model.parameters():
			if param.requires_grad:
				if param.grad is not None:
					param.grad.zero_()
		return

	def forward_pass(self,X,**kwargs):
		""" Do a forward pass through the PyTorch model and return the 
		model outputs (predicted labels). The outputs should be the same shape 
		as the true labels
	
		:param X: model features
		:type X: numpy ndarray

		:return: predictions
		:rtype: torch.Tensor
		"""
		X_torch = torch.tensor(X,requires_grad=True)
		predictions = self.pytorch_model(X_torch.float()).view(-1)
		return predictions

	def backward_pass(self,external_grad):
		""" Do a backward pass through the PyTorch model and return the
		(vector) gradient of the model with respect to theta as a numpy ndarray

		:param external_grad: The gradient of the model with respect to itself
			see: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#differentiation-in-autograd
			for more details
		:type external_grad: torch.Tensor 
		"""
		self.predictions.backward(gradient=external_grad,retain_graph=True)
		grad = torch.cat((self.pytorch_model.bias.grad,self.pytorch_model.weight.grad.view(-1)))
		return grad

class PytorchCNN(SupervisedPytorchBaseModel):
	def __init__(self,input_dim,output_dim):
		""" Implements linear regression using a single Pytorch linear layer

		:param input_dim: Number of features
		:param output_dim: Size of output layer (number of label columns)
		"""
		super().__init__(input_dim,output_dim)
		self.param_sizes = self.get_param_sizes()

	def create_model(self,**kwargs):
		""" Create the pytorch model and return it
		Inputs are N,1,28,28 where N is the number of them,
		1 channel and 28x28 pixels.
		Do Conv2d,ReLU,maxpool twice then
		output in a fully connected layer to 10 output classes
		"""
		cnn = nn.Sequential(         
			nn.Conv2d(
				in_channels=1,              
				out_channels=16,            
				kernel_size=5,              
				stride=1,                   
				padding=2,                  
			),                              
			nn.ReLU(),                      
			nn.MaxPool2d(kernel_size=2),   
			nn.Conv2d(16, 32, 5, 1, 2),     
			nn.ReLU(),                      
			nn.MaxPool2d(2),
			nn.Flatten(),
			nn.Linear(32 * 7 * 7, 10),
			nn.Softmax(dim=1)
		)       
		return cnn

	def get_param_sizes(self):
		param_sizes = []
		for param in self.pytorch_model.parameters():
			if param.requires_grad:
				param_sizes.append(param.numel())
		return param_sizes

	def get_initial_weights(self,*args):
		""" Return initial weights as a flattened 1D array
		Also return the number of elements in each model parameter """
		layer_params_list = []
		for param in self.pytorch_model.parameters():
			if param.requires_grad:
				param_numpy = param.detach().numpy()
				layer_params_list.append(param_numpy.flatten())
		return np.concatenate(layer_params_list)

	def update_model_params(self,theta,**kwargs):
		""" Update all model parameters using theta, which must be reshaped

		:param theta: model weights
		:type theta: numpy ndarray
		"""
		# Update model parameters using flattened array
		with torch.no_grad():
			i = 0
			startindex = 0
			for param in self.pytorch_model.parameters():
				if param.requires_grad:
					nparams = self.param_sizes[i]
					param_shape = param.shape
					theta_numpy = theta[startindex:startindex+nparams]
					theta_torch = torch.from_numpy(theta_numpy).view(param_shape)
					param.copy_(theta_torch)
					i+=1
					startindex+=nparams
		return

	def zero_gradients(self):
		""" Zero out gradients of all model parameters """
		for param in self.pytorch_model.parameters():
			if param.requires_grad:
				if param.grad is not None:
					param.grad.zero_()
		return

	def forward_pass(self,X,**kwargs):
		""" Do a forward pass through the PyTorch model and return the 
		model outputs (predicted labels). The outputs should be the same shape 
		as the true labels
	
		:param X: model features
		:type X: numpy ndarray

		:return: predictions
		:rtype: torch.Tensor
		"""
		X_torch = torch.tensor(X,requires_grad=True)
		predictions = self.pytorch_model(X_torch.float())
		return predictions

	def backward_pass(self,external_grad):
		""" Do a backward pass through the PyTorch model and return the
		(vector) gradient of the model with respect to theta as a numpy ndarray

		:param external_grad: The gradient of the model with respect to itself
			see: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#differentiation-in-autograd
			for more details
		:type external_grad: torch.Tensor 
		"""
		self.predictions.backward(gradient=external_grad,retain_graph=True)
		grad_params_list = []
		for param in self.pytorch_model.parameters():
			if param.requires_grad:
				grad_numpy = param.grad.numpy()
				grad_params_list.append(grad_numpy.flatten())
		return np.concatenate(grad_params_list)