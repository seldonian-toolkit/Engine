# tensorflow_cnn.py

from seldonian.models.tensorflow_model import SupervisedTensorFlowBaseModel

import tensorflow as tf

class GenderClassifierCNN(SupervisedTensorFlowBaseModel):
	def __init__(self,**kwargs):
		""" Base class for Supervised learning Seldonian
		models implemented in TensorFlow
		 
		"""
		super().__init__()

	def create_model(self,**kwargs):
		""" Create the TensorFlow model and return it
		"""
		input_shape=(48,48,1)
		cnn = tf.keras.Sequential([
		    tf.keras.layers.InputLayer(input_shape=input_shape,),
		    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
		    tf.keras.layers.BatchNormalization(),
		    tf.keras.layers.MaxPooling2D((2, 2)),
		    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
		    tf.keras.layers.MaxPooling2D((2, 2)),
		    tf.keras.layers.Flatten(),
		    tf.keras.layers.Dense(64, activation='relu'),
		    tf.keras.layers.Dropout(rate=0.5,),
		    tf.keras.layers.Dense(1, activation='sigmoid',)
		])
		cnn.build(input_shape=input_shape)
		return cnn

	def forward_pass(self,X,**kwargs):
		""" Do a forward pass through the Tensorflow model and return the 
		model outputs (predicted labels). The outputs should be the same shape 
		as the true labels
	
		:param X: model features
		:type X: numpy ndarray

		:return: predictions
		:rtype: torch.Tensor
		"""
		with tf.GradientTape(persistent=True) as tape:
			X_tf = tf.convert_to_tensor(X)
			predictions = self.tensorflow_model(X_tf)
			res = tf.squeeze(predictions)
		self.tape = tape
		return res

