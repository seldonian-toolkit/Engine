# tensorflow_model.py

import autograd.numpy as np  # Thinly-wrapped version of Numpy
from autograd.extend import primitive, defvjp
from seldonian.models.models import SupervisedModel

import tensorflow as tf

@primitive
def tf_predict(theta, X, model, **kwargs):
    """Do a forward pass through the TensorFlow model.
    Must convert back to numpy array before returning

    :param theta: model weights
    :type theta: numpy ndarray
    :param X: model features
    :type X: numpy ndarray

    :param model: An instance of a class inheriting from
            SupervisedTensorFlowBaseModel

    :return pred_numpy: model predictions
    :rtype pred_numpy: numpy ndarray same shape as labels
    """
    # First update model weights
    model.update_model_params(theta, **kwargs)
    # Do the forward pass
    pred = model.forward_pass(X, **kwargs)
    # set the predictions attribute of the model
    model.predictions = pred
    # Convert predictions into a numpy array
    pred_numpy = pred.numpy()
    return pred_numpy

def tf_predict_vjp(ans, theta, X, model):
    """Do a backward pass through the TensorFlow model,
    obtaining the Jacobian d pred / dtheta.
    Must convert back to numpy array before returning

    :param ans: The result from tf_predict
    :type ans: numpy ndarray
    :param theta: model weights
    :type theta: numpy ndarray
    :param X: model features
    :type X: numpy ndarray

    :param model: An instance of a class inheriting from
            SupervisedTensorFlowBaseModel

    :return fn: A function representing the vector Jacobian operator
    """

    def fn(v):
        # v is a vector of shape ans, the return value of mypredict()
        # return a 1D array [dF_i/dtheta[0],dF_i/dtheta[1],dF_i/dtheta[2]],
        # where i is the data row index
        # print("v:")
        # print(v)
        # input("next")
        dpred_dtheta = model.backward_pass(v)
        return dpred_dtheta

    return fn

# Link the predict function with its gradient,
# telling autograd not to look inside either of these functions
defvjp(tf_predict, tf_predict_vjp)


class SupervisedTensorFlowBaseModel(SupervisedModel):
    def __init__(self, **kwargs):
        """Base class for Supervised learning Seldonian
        models implemented in TensorFlow

        :param device: The PyTorch device string indicating the
                hardware on which to run the model,
                e.g. "cpu", "cuda", "mps".
        :type device: str
        """
        super().__init__()
        self.tensorflow_model = self.create_model(**kwargs)
        self.param_sizes = self.get_param_sizes()
        self.weights_updated = False

    def predict(self, theta, X, **kwargs):
        """Do a forward pass through the PyTorch model.
        Must convert back to numpy array before returning

        :param theta: model weights
        :type theta: numpy ndarray

        :param X: model features
        :type X: numpy ndarray

        :return pred_numpy: model predictions
        :rtype pred_numpy: numpy ndarray same shape as labels
        """
        return tf_predict(theta, X, self)

    def get_model_params(self, *args):
        """Return initial weights as a flattened 1D array
        Also return the number of elements in each model parameter"""
        layer_params_list = []
        for param in self.tensorflow_model.trainable_weights:
            layer_params_list.append(param.numpy().flatten())
        return np.concatenate(layer_params_list)

    def get_param_sizes(self):
        """Get the sizes (shapes) of each of the model parameters"""
        param_sizes = []
        for param in self.tensorflow_model.trainable_weights:
            param_sizes.append(np.prod(param.shape))
        return param_sizes

    def update_model_params(self, theta, **kwargs):
        """Update all model parameters using theta,
        which must be reshaped

        :param theta: model weights
        :type theta: numpy ndarray
        """
        # Update model parameters using flattened array
        i = 0
        startindex = 0
        for param in self.tensorflow_model.trainable_weights:
            nparams = self.param_sizes[i]
            param_shape = param.shape
            theta_numpy = theta[startindex : startindex + nparams]
            theta_tf_flat = tf.convert_to_tensor(theta_numpy)
            theta_tf = tf.reshape(theta_tf_flat, param_shape)
            param.assign(theta_tf)
            i += 1
            startindex += nparams
        return

    def forward_pass(self, X, **kwargs):
        """Do a forward pass through the TensorFlow model and return the
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
        self.tape = tape
        return predictions

    def backward_pass(self, v):
        """Do a backward pass through the TensorFlow model and return the
        (vector) gradient of the model with respect to theta as a numpy ndarray

        :param external_grad: The gradient of the model with respect to itself
                see: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#differentiation-in-autograd
                for more details
        :type external_grad: torch.Tensor
        """
        grad_params_list = []
        grads = self.tape.gradient(
            self.predictions,
            self.tensorflow_model.trainable_weights,
            output_gradients=v,
        )
        for grad in grads:
            grad_numpy = grad.numpy()
            grad_params_list.append(grad_numpy.flatten())
        return np.concatenate(grad_params_list)

    def create_model(self, **kwargs):
        """Create the TensorFlow model and return it"""
        raise NotImplementedError("Implement this method in child class")
