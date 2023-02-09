from .pytorch_model import SupervisedPytorchBaseModel
import torch.nn as nn

class PytorchCNN(SupervisedPytorchBaseModel):
	def __init__(self,device):
		""" Implements an example CNN with PyTorch. 
		CNN consists of two hidden layers followed 
		by a linear + softmax output layer 

		:param device: The torch device, e.g., 
			"cuda" (NVIDIA GPU), "cpu" for CPU only,
			"mps" (Mac M1 GPU)
		"""
		super().__init__(device)

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