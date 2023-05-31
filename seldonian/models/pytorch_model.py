# pytorch_model.py
### A simple single layer Pytorch model implementing linear regression

import autograd.numpy as np  # Thinly-wrapped version of Numpy
from autograd.extend import primitive, defvjp
from seldonian.models.models import SupervisedModel

import torch
import torch.nn as nn


@primitive
def pytorch_predict(theta, X, model, **kwargs):
    """Do a forward pass through the PyTorch model.
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
    if not model.params_updated:
        model.update_model_params(theta, **kwargs)
        model.params_updated = True
    # Do the forward pass
    pred = model.forward_pass(X, **kwargs)
    # set the predictions attribute of the model

    model.predictions = pred

    # Convert predictions into a numpy array
    pred_numpy = pred.cpu().detach().numpy()
    return pred_numpy


def pytorch_predict_vjp(ans, theta, X, model):
    """Do a backward pass through the PyTorch model,
    obtaining the Jacobian d pred / dtheta.
    Must convert back to numpy array before returning

    :param ans: The result from the forward pass
    :type ans: numpy ndarray
    :param theta: model weights
    :type theta: numpy ndarray
    :param X: model features
    :type X: numpy ndarray

    :param model: An instance of a class inheriting from
            SupervisedPytorchBaseModel

    :return fn: A function representing the vector Jacobian operator
    """
    local_predictions = model.predictions

    def fn(v):
        # v is a vector of shape ans, the return value of mypredict()
        # return a 1D array [dF_i/dtheta[0],dF_i/dtheta[1],dF_i/dtheta[2]],
        # where i is the data row index
        external_grad = torch.from_numpy(v).float().to(model.device)
        dpred_dtheta = model.backward_pass(local_predictions, external_grad)
        model.params_updated = False  # resets for the
        return np.array(dpred_dtheta)

    return fn


# Link the predict function with its gradient,
# telling autograd not to look inside either of these functions
defvjp(pytorch_predict, pytorch_predict_vjp)


class SupervisedPytorchBaseModel(SupervisedModel):
    def __init__(self, device, **kwargs):
        """Base class for Supervised learning Seldonian
        models implemented in Pytorch

        :param device: The PyTorch device string indicating the
                hardware on which to run the model,
                e.g. "cpu", "cuda", "mps".
        :type device: str
        """
        super().__init__()
        self.device = device
        self.pytorch_model = self.create_model(**kwargs)
        self.pytorch_model.to(self.device)
        self.param_sizes = self.get_param_sizes()
        self.params_updated = False

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
        return pytorch_predict(theta, X, self)

    def get_model_params(self, *args):
        """Return weights of the model as a flattened 1D array
        Also return the number of elements in each model parameter"""
        layer_params_list = []
        for param in self.pytorch_model.parameters():
            if param.requires_grad:
                param_numpy = param.cpu().detach().numpy()
                layer_params_list.append(param_numpy.flatten())
        return np.concatenate(layer_params_list)

    def get_param_sizes(self):
        """Get the sizes (shapes) of each of the model parameters"""
        param_sizes = []
        for param in self.pytorch_model.parameters():
            if param.requires_grad:
                param_sizes.append(param.numel())
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
        for param in self.pytorch_model.parameters():
            if param.requires_grad:
                nparams = self.param_sizes[i]
                param_shape = param.shape
                theta_numpy = theta[startindex : startindex + nparams]
                theta_torch = torch.from_numpy(theta_numpy).view(param_shape)
                with torch.no_grad():
                    param.copy_(theta_torch)
                i += 1
                startindex += nparams
        return

    def zero_gradients(self):
        """Zero out gradients of all model parameters"""
        for param in self.pytorch_model.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    param.grad.zero_()
        return

    def forward_pass(self, X, **kwargs):
        """Do a forward pass through the PyTorch model and return the
        model outputs (predicted labels). The outputs should be the same shape
        as the true labels

        :param X: model features
        :type X: numpy ndarray

        :return: predictions
        :rtype: torch.Tensor
        """
        X_torch = torch.tensor(X).float().to(self.device)
        predictions = self.pytorch_model(X_torch)
        return predictions

    def backward_pass(self, predictions, external_grad):
        """Do a backward pass through the PyTorch model and return the
        (vector) gradient of the model with respect to theta as a numpy ndarray

        :param external_grad: The gradient of the model with respect to itself
                see: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#differentiation-in-autograd
                for more details
        :type external_grad: torch.Tensor
        """
        self.zero_gradients()
        predictions.backward(gradient=external_grad,retain_graph=True)
        grad_params_list = []
        for param in self.pytorch_model.parameters():
            if param.requires_grad:
                grad_numpy = param.grad.cpu().numpy()
                grad_params_list.append(grad_numpy.flatten())

        return np.concatenate(grad_params_list)

    def create_model(self, **kwargs):
        """Create the pytorch model and return it"""
        raise NotImplementedError("Implement this method in child class")
