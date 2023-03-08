import autograd.numpy as np

from .sklearn_model import SupervisedSkLearnBaseModel
from sklearn.linear_model import LinearRegression


class SkLearnLinearRegressor(SupervisedSkLearnBaseModel):
    def __init__(self, **kwargs):
        """Implements a linear regressor in Scikit-learn"""
        self.has_intercept = True
        super().__init__(**kwargs)

    def create_model(self, **kwargs):
        """Create the scikit-learn linear regressor"""
        model = LinearRegression(**kwargs)
        return model

    def forward_pass(self, X):
        """Make predictions given features"""
        predictions = self.sklearn_model.predict(X)
        return predictions

    def backward_pass(self, theta, X):
        """Return the Jacobian d(forward_pass)_i/dtheta_{j+1},
        where i run over datapoints and j run over model parameters,
        keeping in mind that there is a y-intercept term (hence the j+1 not j)
        in the predict function
        """
        m = len(X)
        X_withintercept = np.hstack([np.ones((m, 1)), np.array(X)])
        return X_withintercept
