""" Main module containing Seldonian machine learning models """
import autograd.numpy as np  # Thinly-wrapped version of Numpy

from sklearn.linear_model import LinearRegression, LogisticRegression


class SeldonianModel(object):
    def __init__(self):
        """Parent class for all machine learning models"""
        pass


class SupervisedModel(SeldonianModel):
    def __init__(self):
        """Parent class for all supervised learning models"""
        super().__init__()


class RegressionModel(SupervisedModel):
    def __init__(self):
        """Parent class for all regression-based models"""
        super().__init__()
        self.has_intercept = True

    def predict(self):
        raise NotImplementedError("Implement this method in child class")


class LinearRegressionModel(RegressionModel):
    def __init__(self):
        """Implements linear regression"""
        super().__init__()
        self.model_class = LinearRegression

    def predict(self, theta, X):
        """Predict label using the linear model

        :param theta: The parameter weights
        :type theta: numpy ndarray
        :param X: The features
        :type X: numpy ndarray
        :return: predicted labels
        :rtype: numpy ndarray
        """
        return theta[0] + (X @ theta[1:])

    def fit(self, X, Y):
        """Train the model using the feature,label pairs

        :param X: features
        :type X: NxM numpy ndarray
        :param Y: labels
        :type Y: Nx1 numpy ndarray
        :return: weights from the fitted model
        :rtype: numpy ndarray
        """
        reg = self.model_class().fit(X, Y)
        return np.hstack([reg.intercept_, reg.coef_])


class LinearRegressionModelListFeatures(RegressionModel):
    def __init__(self):
        """Implements linear regression"""
        super().__init__()
        self.model_class = LinearRegression

    def predict(self, theta, X):
        """Predict label using the linear model

        :param theta: The parameter weights
        :type theta: numpy ndarray
        :param X: The features
        :type X: numpy ndarray
        :return: predicted labels
        :rtype: numpy ndarray
        """
        X_array = np.hstack(X)
        return theta[0] + (X_array @ theta[1:])

    def fit(self, X, Y):
        """Train the model using the feature,label pairs

        :param X: features
        :type X: NxM numpy ndarray
        :param Y: labels
        :type Y: Nx1 numpy ndarray
        :return: weights from the fitted model
        :rtype: numpy ndarray
        """
        X_array = np.hstack(X)
        reg = self.model_class().fit(X_array, Y)
        return np.hstack([reg.intercept_, reg.coef_])


class BoundedLinearRegressionModel(LinearRegressionModel):
    def __init__(self):
        """Implements linear regression
        with a bounded predict function.
        Overrides several parent methods.
        Assumes y-intercept is 0."""
        super().__init__()

    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def predict(self, theta, X):
        """Overrides the original predict
        function to squash predictions

        :param theta: The parameter weights
        :type theta: numpy ndarray
        :param X: The features
        :type X: numpy ndarray
        :return: predicted labels
        :rtype: numpy ndarray
        """
        y_min, y_max = -3, 3
        # Want range of Y_hat to be twice that of Y
        # and want size of interval on either side of Y_min and Y_max
        # to be the same. The unique solution to this is:
        s = 2.0  # 1 gives you the same bound size as y
        y_hat_min = y_min * (1 + s) / 2 + y_max * (1 - s) / 2
        y_hat_max = y_max * (1 + s) / 2 + y_min * (1 - s) / 2
        Z = theta[0] + (X @ theta[1:])
        return self._sigmoid(Z) * (y_hat_max - y_hat_min) + y_hat_min


class ClassificationModel(SupervisedModel):
    def __init__(self):
        """Parent class for all classification-based
        machine learning models

        Currently only supports binary classification
        """
        super().__init__()

    def predict(self):
        raise NotImplementedError("Implement this method in child class")


class BaseLogisticRegressionModel(ClassificationModel):
    def __init__(self):
        """Base class for binary and multi-class
        logistic regression"""
        super().__init__()
        self.model_class = LogisticRegression
        self.has_intercept = True

    def fit(self, X, Y):
        """Train the model using features and labels.
        Let:
                i = number of datapoints
                j = number of features (including bias term, if provied)
                k = number of classes

        :param X: The features
        :type X: array of shape (i,j)
        :param Y: The labels
        :type Y: array of shape (i,k)
        :return: fitted model weights
        :rtype: array of shape (j,k)
        """
        reg = self.model_class().fit(X, Y)
        theta = np.squeeze(np.vstack([reg.intercept_, reg.coef_.T]))
        return theta


class BinaryLogisticRegressionModel(BaseLogisticRegressionModel):
    def __init__(self):
        """Implements binary logistic regression"""
        super().__init__()

    def predict(self, theta, X):
        """Predict the probability of
        having the positive class label for each data point
        in X. Let:
                i = number of datapoints
                j = number of features (including bias term, if provied)

        :param theta: The parameter weights
        :type theta: array of length j or shape (j,1)
        :param X: The features
        :type X: array of shape (i,j)
        :return: predictions for each class each observation
        :rtype: array of length i or shape (i,1)
        """
        Z = theta[0] + (X @ theta[1:])  # (i,j) x (j,1) -> (i,1)
        Y_pred = 1 / (1 + np.exp(-Z))
        return Y_pred


class MultiClassLogisticRegressionModel(BaseLogisticRegressionModel):
    def __init__(self):
        """Implements multi-class
        logistic regression"""
        super().__init__()

    def predict(self, theta, X):
        """Predict the probability of
        having each class label for each data point
        in X. Let:
                i = number of datapoints
                j = number of features (including bias term, if provied)
                k = number of classes

        :param theta: The parameter weights
        :type theta: array of shape (j,k)
        :param X: The features
        :type X: array of shape (i,j)
        :return: predictions for each class each observation
        :rtype: array of shape (i x k)
        """
        Z = theta[0] + (X @ theta[1:])  # (i,j) x (j,k) -> (i,k)
        # softmax to get probabilites
        Y_pred = np.exp(Z) / np.sum(np.exp(Z), axis=-1, keepdims=True)

        return Y_pred


class DummyClassifierModel(ClassificationModel):
    def __init__(self):
        """Implements a classifier that always predicts
        the positive class, regardless of input"""
        super().__init__()
        self.has_intercept = False

    def predict(self, theta, X):
        """Predict the probability of
        having the positive class label

        :param theta: The parameter weights
        :type theta: numpy ndarray
        :param X: The features
        :type X: numpy ndarray
        :return: predictions for each observation
        :rtype: float
        """

        return np.ones(len(X))


class RandomClassifierModel(ClassificationModel):
    def __init__(self):
        """Implements a classifier that always predicts
        that the positive class has prob=0.5,
        regardless of input"""
        super().__init__()
        self.has_intercept = False

    def predict(self, theta, X):
        """Predict the probability of
        having the positive class label

        :param theta: The parameter weights
        :type theta: numpy ndarray
        :param X: The features
        :type X: numpy ndarray
        :return: predictions for each observation
        :rtype: float
        """
        return 0.5 * np.ones(len(X))
