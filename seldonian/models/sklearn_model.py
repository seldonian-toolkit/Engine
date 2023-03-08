# skelarn_model.py
### Base model for scikit-learn models

import autograd.numpy as np  # Thinly-wrapped version of Numpy
from autograd.extend import primitive, defvjp
from seldonian.models.models import SupervisedModel


@primitive
def sklearn_predict(theta, X, model, **kwargs):
    """Do a forward pass through the sklearn model.
    Must convert back to numpy array before returning

    :param theta: model weights
    :type theta: numpy ndarray
    :param X: model features
    :type X: numpy ndarray

    :param model: An instance of a class inheriting from
            SupervisedSkLearnBaseModel

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

    # Predictions must be a numpy array

    return pred


def sklearn_predict_vjp(ans, theta, X, model):
    """Do a backward pass through the Sklearn model,
    obtaining the Jacobian d pred / dtheta.
    Must convert back to numpy array before returning

    :param ans: The result from the forward pass
    :type ans: numpy ndarray
    :param theta: model weights
    :type theta: numpy ndarray
    :param X: model features
    :type X: numpy ndarray

    :param model: An instance of a class inheriting from
            SupervisedSkLearnBaseModel

    :return fn: A function representing the vector Jacobian operator
    """

    def fn(v):
        # v is a vector of shape ans, the return value of mypredict()
        # This function returns a 1D array:
        # [dF_i/dtheta[0],dF_i/dtheta[1],dF_i/dtheta[2],...],
        # where i is the data row index
        dpred_dtheta = model.backward_pass(theta, X)
        model.params_updated = False  # resets for the next forward pass
        return v.T @ dpred_dtheta

    return fn


# Link the predict function with its gradient,
# telling autograd not to look inside either of these functions
defvjp(sklearn_predict, sklearn_predict_vjp)


class SupervisedSkLearnBaseModel(SupervisedModel):
    def __init__(self, **kwargs):
        """Base class for Supervised learning Seldonian
        models implemented in scikit-learn
        """
        super().__init__()
        self.sklearn_model = self.create_model(**kwargs)
        self.params_updated = False

    def predict(self, theta, X, **kwargs):
        """Do a forward pass through the sklearn model.
        Must convert back to numpy array before returning

        :param theta: model weights
        :type theta: numpy ndarray

        :param X: model features
        :type X: numpy ndarray

        :return pred_numpy: model predictions
        :rtype pred_numpy: numpy ndarray same shape as labels
        """
        return sklearn_predict(theta, X, self)

    def get_model_params(self, *args):
        """Return weights of the model as a flattened 1D array"""
        if self.has_intercept:
            return np.hstack([self.sklearn_model.intercept_, self.sklearn_model.coef_])
        else:
            return self.sklearn_model.coef_  # automatically a numpy array

    def update_model_params(self, theta, **kwargs):
        """Update all model parameters using theta,
        which must be reshaped

        :param theta: model weights
        :type theta: numpy ndarray
        """
        # Update model parameters using flattened array
        if self.has_intercept:
            self.sklearn_model.__setattr__("intercept_", theta[0])
            self.sklearn_model.__setattr__("coef_", theta[1:])
        else:
            self.sklearn_model.__setattr__("coef_", theta)

    def forward_pass(self, X, **kwargs):
        """Do a forward pass through the Sklearn model and return the
        model outputs (predicted labels or probabilities,
        depending on the model). Here, a forward pass
        is just a call to self.sklearn_model.predict()

        :param X: model features
        :type X: numpy ndarray

        :return: predictions
        :rtype: numpy ndarray
        """
        raise NotImplementedError("Implement this method in child class")

    def backward_pass(self, predictions, external_grad):
        """Do a backward pass through the model and return the
        (vector) gradient of the model with respect to theta as a numpy ndarray

        """
        raise NotImplementedError("Implement this method in child class")

    def create_model(self, **kwargs):
        """Create the sklearn model and return it"""
        raise NotImplementedError("Implement this method in child class")
