from seldonian.models.pytorch_model import SupervisedPytorchBaseModel
import torch.nn as nn
import torch

class FacialRecogCNNModel(nn.Module):
	def __init__(self):
		super(FacialRecogCNNModel, self).__init__()
		# Define all layers here
		self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
		self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
		self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
		self.cnn4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
		
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(kernel_size=2)
		
		self.Batch1=nn.BatchNorm2d(16)
		self.Batch2=nn.BatchNorm2d(32)
		self.Batch3=nn.BatchNorm2d(64)
		self.Batch4=nn.BatchNorm2d(128)
		
		# Fully connected 1 (readout)
		self.fc1 = nn.Linear(128 * 1 * 1, 128) 
		self.fc2=nn.Linear(128,256)
		self.fc3=nn.Linear(256,2)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
 		# Call all layers here
		out = self.cnn1(x) 
		out = self.relu(out)
		out = self.maxpool(out)
		out=self.Batch1(out)

		out = self.cnn2(out)
		out = self.relu(out)
		out = self.maxpool(out)
		out=self.Batch2(out)
		
		out = self.cnn3(out)
		out = self.relu(out)
		out = self.maxpool(out)
		out=self.Batch3(out)
		
		out = self.cnn4(out)
		out = self.relu(out)
		out = self.maxpool(out)
		out=self.Batch4(out)
		
		# Resize
		# Original size: (100, 32, 7, 7)
		# New out size: (100, 32*7*7)
		out = torch.flatten(out,start_dim=1)

		# Linear functions (readout)
		out = self.fc1(out)
		out=self.fc2(out)
		out=self.fc3(out)

		# Softmax to make probabilities
		out=self.softmax(out)[:,1] 

		return out

class PytorchFacialRecog(SupervisedPytorchBaseModel):
	def __init__(self,device):
		""" Implements a CNN with PyTorch. 
		CNN consists of two hidden layers followed 
		by a linear + softmax output layer 

		:param input_dim: Number of features
		:param output_dim: Size of output layer (number of label columns)
		"""
		super().__init__(device)

	def create_model(self,**kwargs):
		""" Create the pytorch model and return it
		Inputs are N,1,28,28 where N is the number of them,
		1 channel and 28x28 pixels.
		Do Conv2d,ReLU,maxpool twice then
		output in a fully connected layer to 10 output classes
		"""
		return FacialRecogCNNModel()