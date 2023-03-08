import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch import optim
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

from full_cnn import CNNModelNoSoftmax, CNNHeadlessModel

from seldonian.utils.io_utils import load_pickle,save_pickle

def train(num_epochs, cnn, loaders):
	cnn.train()
		
	# Train the model
	total_step = len(loaders['candidate'])
		
	for epoch in range(num_epochs):
		for i, (images, labels) in enumerate(loaders['candidate']):

			images = images.to(device)
			labels = labels.to(device)
			b_x = Variable(images)   # batch x
			output = cnn(b_x)
			b_y = Variable(labels)   # batch y
			loss = loss_func(output, b_y)
			
			# clear gradients for this training step   
			optimizer.zero_grad()           
			
			# backpropagation, compute gradients 
			loss.backward()    
			# apply gradients             
			optimizer.step()                
			
			if (i+1) % 100 == 0:
				print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
					   .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

def test(cnn):
	cnn.eval()
	test_loss = 0
	correct = 0
	test_loader = loaders['safety']
	with torch.no_grad():
		for images, target in test_loader:
			images = images.to(device)
			target = target.to(device)
			output = cnn(images)
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
	correct, len(test_loader.dataset),
	100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
	# Writing this as a script so it is more reproducible than Jupyter Notebook
	# Model does not have dropout layers and does not have softmax so that it can 
	# be trained using PyTorch objective

	batch_size = 100
	num_epochs = 10
	torch.manual_seed(0)
	device = torch.device("mps")

	feat_f = './features.pkl'
	label_f = './labels.pkl'
	print("loading data...")
	features = load_pickle(feat_f)
	labels = load_pickle(label_f)
	print("done")

	print("converting data to tensors")
	features_cand = features[:11850]
	features_safety = features[11850:]
	labels_cand = labels[:11850]
	labels_safety = labels[11850:]

	features_cand_tensor = torch.from_numpy(features_cand)
	features_safety_tensor = torch.from_numpy(features_safety)
	labels_cand_tensor = torch.from_numpy(labels_cand)
	labels_safety_tensor = torch.from_numpy(labels_safety)
	print("done")
	

	print("Making data loaders...")
	candidate_dataset=torch.utils.data.TensorDataset(
		features_cand_tensor,labels_cand_tensor) 
	candidate_dataloader=torch.utils.data.DataLoader(
		candidate_dataset,batch_size=batch_size,shuffle=False) 

	safety_dataset=torch.utils.data.TensorDataset(
		features_safety_tensor,labels_safety_tensor) 
	safety_dataloader=torch.utils.data.DataLoader(
		safety_dataset,batch_size=batch_size,shuffle=False) 

	loaders = {
		'candidate' : candidate_dataloader,
		'safety'  : safety_dataloader
	}
	print("done")

	print("Building model and putting it on the GPU")
	cnn = CNNModelNoSoftmax()
	cnn.to(device)
	print("done")

	learning_rate=0.001

	# Loss and optimizer
	loss_func = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

	print("Check state dict before training so we can compare to after training")
	sd_before_training = cnn.state_dict()
	print(sd_before_training['cnn1.weight'][0])
	print("done.\n")

	print(f"Training model on full CNN with {num_epochs} epochs")
	train(num_epochs, cnn, loaders)
	print("done.\n")

	print("Evaluating model:")
	test(cnn)

	print("Compare state dict after training to verify parameters were changed")
	sd_after_training = cnn.state_dict()
	print(sd_after_training['cnn1.weight'][0])

	print("Putting headless model on GPU")
	cnn_headless = CNNHeadlessModel().to(device)
	print("done")

	print("Loading state dictionary into headless model...")
	del sd_after_training['fc3.weight']
	del sd_after_training['fc3.bias']
	cnn_headless.load_state_dict(sd_after_training)
	print("done.")

	print("Verify that the weights were copied over to the headless model:")
	sd_headless = cnn_headless.state_dict()
	print(sd_headless['cnn1.weight'][0])

	print("passing all train and test images through the headless model to create latent features...")
	new_features = np.zeros((23700,256))
	new_labels = np.zeros(23700)
	for i,(images, labels) in enumerate(loaders['candidate']):
		start_index = i*batch_size
		end_index = start_index + len(images)
		images = images.to(device)
		new_labels[start_index:end_index] = labels.numpy()
		new_features[start_index:end_index] = cnn_headless(images).cpu().detach().numpy()
	for j,(images, labels) in enumerate(loaders['safety']):
		start_index = end_index
		end_index = start_index + len(images)
		images = images.to(device)
		new_labels[start_index:end_index] = labels.numpy()
		new_features[start_index:end_index] = cnn_headless(images).cpu().detach().numpy()
	print("done.")

	print("Make sure there are some non-zero values in features. ")
	print(new_features[1001])

	print("Saving latent features and labels")
	save_pickle('facial_gender_latent_features.pkl',new_features)
	save_pickle('facial_gender_labels.pkl',new_labels)
	print("done.")