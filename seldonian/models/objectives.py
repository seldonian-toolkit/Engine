""" Objective functions """

import autograd.numpy as np  # Thinly-wrapped version of Numpy
import math

from seldonian.utils.stats_utils import (weighted_sum_gamma,
    custom_cumprod, stability_const)


""" Regression """

def Mean_Squared_Error(model, theta, X, Y, **kwargs):
    """
    Calculate mean squared error over the whole sample

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Sample mean squared error
    :rtype: float
    """
    n = len(Y)  # Y guaranteed to be a numpy array, X isn't.
    prediction = model.predict(theta, X)  # vector of values
    res = sum(pow(prediction - Y, 2)) / n

    return res


def gradient_Mean_Squared_Error(model, theta, X, Y, **kwargs):
    """Gradient of the mean squared error w.r.t. theta

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Sample mean squared error
    :rtype: float
    """
    if type(X) == list:
        raise NotImplementedError(
            "This function is not supported when features are in a list. "
            "Convert features to a numpy array if possible or use autodiff "
            " to get the gradient."
        )
    n = len(Y)
    prediction = model.predict(theta, X)  # vector of values
    err = prediction - Y
    X_withintercept = np.hstack([np.ones((n, 1)), np.array(X)])
    return 2 / n * np.dot(err, X_withintercept)


def Mean_Error(model, theta, X, Y, **kwargs):
    """
    Calculate mean error (y_hat-y) over the whole sample

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Sample mean squared error
    :rtype: float
    """
    n = len(X)
    prediction = model.predict(theta, X)  # vector of values
    res = sum(prediction - Y) / n
    return res


def gradient_Bounded_Squared_Error(model, theta, X, Y, **kwargs):
    """Analytical gradient of the bounded squared error
    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: the gradient evaluated at this theta
    :rtype: float
    """
    n = len(X)
    y_min, y_max = -3, 3
    # Want range of Y_hat to be twice that of Y
    # and want size of interval on either side of Y_min and Y_max
    # to be the same. The unique solution to this is:
    s = 1.5
    y_hat_min = y_min * (1 + s) / 2 + y_max * (1 - s) / 2
    y_hat_max = y_max * (1 + s) / 2 + y_min * (1 - s) / 2

    c1 = y_hat_max - y_hat_min
    c2 = -y_hat_min

    Y_hat = model.predict(theta, X)  # vector of values
    Y_hat_old = (Y_hat - y_hat_min) / (y_hat_max - y_hat_min)
    sig = model._sigmoid(Y_hat_old)

    term1 = Y - (c1 * sig - c2)
    term2 = -c1 * sig * (1 - sig) * X[:, 0]
    s = sum(term1 * term2)
    return -2 / n * s


""" Classification """


def binary_logistic_loss(model, theta, X, Y, **kwargs):
    """Calculate average logistic loss
    over all data points for binary classification

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: logistic loss
    :rtype: float
    """
    Y_pred = model.predict(theta, X)
    # Add stability constant. This guards against
    # predictions that are 0 or 1, which cause log(Y_pred) or
    # log(1.0-Y_pred) to be nan. If Y==0 and Y_pred == 1,
    # cost will be np.log(1e-15) ~ -34.
    # Similarly if Y==1 and Y_pred == 0.
    # It's a ceiling in the cost function, essentially.
    res = np.mean(
        -Y * np.log(Y_pred + stability_const)
        - (1.0 - Y) * np.log(1.0 - Y_pred + stability_const)
    )
    return res


def gradient_binary_logistic_loss(model, theta, X, Y, **kwargs):
    """Gradient of binary logistic loss w.r.t. theta

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray
    :return: perceptron loss
    :rtype: float
    """
    h = model.predict(theta, X)
    X_withintercept = np.hstack([np.ones((len(X), 1)), np.array(X)])
    res = (1 / len(X)) * np.dot(X_withintercept.T, (h - Y))
    return res


def multiclass_logistic_loss(model, theta, X, Y, **kwargs):
    """Calculate average logistic loss
    over all data points for multi-class classification

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: logistic loss
    :rtype: float
    """
    # Negative log likelihood
    # In the multi-class setting, y_pred is an i x k matrix
    # where i is the number of samples and k is the number of classes
    # Each entry is the probability of predicting the kth class
    # for the ith sample. We need to get the probability of predicting
    # the true class for each sample and then take the sum of the
    # logs of that.
    Y_pred = model.predict(theta, X)
    N = len(Y)
    probs_trueclasses = Y_pred[np.arange(N), Y.astype("int")]
    return -1 / N * sum(np.log(probs_trueclasses))


def Positive_Rate(model, theta, X, Y, **kwargs):
    """
    Calculate positive rate
    for the whole sample.
    This is the sum of probability of each
    sample being in the positive class
    normalized to the number of predictions

    :param model: SeldonianModel instance:param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray

    :return: Positive rate for whole sample
    :rtype: float between 0 and 1
    """
    if "class_index" in kwargs:
        return _Positive_Rate_multiclass(
            model, theta, X, Y, class_index=kwargs["class_index"]
        )
    else:
        return _Positive_Rate_binary(model, theta, X, Y)


def _Positive_Rate_binary(model, theta, X, Y, **kwargs):
    prediction = model.predict(theta, X)
    return np.sum(prediction) / len(X)  # if all 1s then PR=1.


def _Positive_Rate_multiclass(model, theta, X, Y, class_index, **kwargs):
    prediction = model.predict(theta, X)
    return np.sum(prediction[:, class_index]) / len(X)  # if all 1s then PR=1.


def Negative_Rate(model, theta, X, Y, **kwargs):
    """
    Calculate negative rate
    for the whole sample.
    This is the sum of the probability of each
    sample being in the negative class, which is
    1.0 - probability of being in positive class

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray

    :return: Negative rate for whole sample
    :rtype: float between 0 and 1
    """
    if "class_index" in kwargs:
        return _Negative_Rate_multiclass(
            model, theta, X, Y, class_index=kwargs["class_index"]
        )
    else:
        return _Negative_Rate_binary(model, theta, X, Y)


def _Negative_Rate_binary(model, theta, X, Y, **kwargs):
    # Average probability of predicting the negative class
    prediction = model.predict(theta, X)
    return np.sum(1.0 - prediction) / len(X)  # if all 1s then PR=1.


def _Negative_Rate_multiclass(model, theta, X, Y, class_index, **kwargs):
    # Average probability of predicting class!=class_index
    prediction = model.predict(theta, X)
    return np.sum(1.0 - prediction[:, class_index]) / len(X)


def False_Positive_Rate(model, theta, X, Y, **kwargs):
    """
    Calculate probabilistic average false positive rate
    over the whole sample. The is the average probability of
    predicting the positive class when the true label was
    the negative class.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray

    :return: Average false positive rate
    :rtype: float between 0 and 1
    """
    if "class_index" in kwargs:
        return _False_Positive_Rate_multiclass(
            model, theta, X, Y, class_index=kwargs["class_index"]
        )
    else:
        return _False_Positive_Rate_binary(model, theta, X, Y)


def _False_Positive_Rate_binary(model, theta, X, Y, **kwargs):
    # Average probability of predicting positive class
    # subject to the truth being the other class
    prediction = model.predict(theta, X)
    neg_mask = Y != 1.0
    return np.sum(prediction[neg_mask]) / len(X[neg_mask])


def _False_Positive_Rate_multiclass(model, theta, X, Y, class_index, **kwargs):
    # Sum the probability of predicting class=class_index
    # subject to the true label being any other class
    prediction = model.predict(theta, X)

    neg_mask = Y != class_index
    return np.sum(prediction[:, class_index][neg_mask]) / len(X[neg_mask])


def False_Negative_Rate(model, theta, X, Y, **kwargs):
    """
    Calculate probabilistic average false negative rate
    over the whole sample. The is the average probability
    of predicting the negative class when truth was positive class.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray

    :return: Average false negative rate
    :rtype: float between 0 and 1
    """
    if "class_index" in kwargs:
        return _False_Negative_Rate_multiclass(
            model, theta, X, Y, class_index=kwargs["class_index"]
        )
    else:
        return _False_Negative_Rate_binary(model, theta, X, Y)


def _False_Negative_Rate_binary(model, theta, X, Y, **kwargs):
    # Average probability of being in negative class
    # subject to the truth being the positive class
    prediction = model.predict(theta, X)
    pos_mask = Y == 1.0
    return np.sum(1.0 - prediction[pos_mask]) / len(X[pos_mask])


def _False_Negative_Rate_multiclass(model, theta, X, Y, class_index, **kwargs):
    # Average probability of not having class=class_index
    # subject to the truth being class=class_index
    prediction = model.predict(theta, X)
    pos_mask = Y == class_index
    return np.sum(1.0 - prediction[:, class_index][pos_mask]) / len(X[pos_mask])


def True_Positive_Rate(model, theta, X, Y, **kwargs):
    """
    Calculate true positive rate
    for the whole sample.

    The is the sum of the probability of each
    sample being in the positive class when in fact it was in
    the positive class.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray

    :return: False positive rate for whole sample
    :rtype: float between 0 and 1
    """
    if "class_index" in kwargs:
        return _True_Positive_Rate_multiclass(
            model, theta, X, Y, class_index=kwargs["class_index"]
        )
    else:
        return _True_Positive_Rate_binary(model, theta, X, Y)


def _True_Positive_Rate_binary(model, theta, X, Y, **kwargs):
    # Average probability of predicting the positive class
    # subject to the true label being the positive class
    prediction = model.predict(theta, X)
    pos_mask = Y == 1.0
    return np.sum(prediction[pos_mask]) / len(X[pos_mask])


def _True_Positive_Rate_multiclass(model, theta, X, Y, class_index, **kwargs):
    # Average probability of predicting class=class_index
    # subject to the true label having class=class_index
    prediction = model.predict(theta, X)
    pos_mask = Y == class_index
    return np.sum(prediction[:, class_index][pos_mask]) / len(X[pos_mask])


def True_Negative_Rate(model, theta, X, Y, **kwargs):
    """
    Calculate true negative rate
    for the whole sample.

    The is the sum of the probability of each
    sample being in the negative class when in fact it was in
    the negative class.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray

    :return: False positive rate for whole sample
    :rtype: float between 0 and 1
    """
    if "class_index" in kwargs:
        return _True_Negative_Rate_multiclass(
            model, theta, X, Y, class_index=kwargs["class_index"]
        )
    else:
        return _True_Negative_Rate_binary(model, theta, X, Y)


def _True_Negative_Rate_binary(model, theta, X, Y, **kwargs):
    # Average probability of being in negative class
    # subject to the truth being the negative class
    prediction = model.predict(theta, X)
    neg_mask = Y != 1.0
    return np.sum(1.0 - prediction[neg_mask]) / len(X[neg_mask])


def _True_Negative_Rate_multiclass(model, theta, X, Y, class_index, **kwargs):
    # Average probability of class!=class_index
    # subject to the truth being class!=class_index
    prediction = model.predict(theta, X)
    neg_mask = Y != class_index
    return np.sum(1.0 - prediction[:, class_index][neg_mask]) / len(X[neg_mask])


def True_Positive_Rate(model, theta, X, Y, **kwargs):
    """
    Calculate true positive rate
    for the whole sample.

    The is the sum of the probability of each
    sample being in the positive class when in fact it was in
    the positive class.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray

    :return: False positive rate for whole sample
    :rtype: float between 0 and 1
    """
    if "class_index" in kwargs:
        return _True_Positive_Rate_multiclass(
            model, theta, X, Y, class_index=kwargs["class_index"]
        )
    else:
        return _True_Positive_Rate_binary(model, theta, X, Y)


def Accuracy(model, theta, X, Y, **kwargs):
    """
    Calculate true negative rate
    for the whole sample.

    The is the sum of the probability of each
    sample being in the negative class when in fact it was in
    the negative class.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray

    :return: False positive rate for whole sample
    :rtype: float between 0 and 1
    """
    if kwargs["sub_regime"] == "multiclass_classification":
        return _Accuracy_multiclass(model, theta, X, Y, **kwargs)
    else:
        return _Accuracy_binary(model, theta, X, Y, **kwargs)


def _Accuracy_binary(model, theta, X, Y, **kwargs):
    """Calculate accuracy
    over all data points for binary classification

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: accuracy
    :rtype: float
    """
    n = len(X)
    Y_pred_probs = model.predict(theta, X)
    v = np.where(Y != 1, 1.0 - Y_pred_probs, Y_pred_probs)
    return np.sum(v) / n


def _Accuracy_multiclass(model, theta, X, Y, **kwargs):
    """Calculate accuracy
    over all data points for multi-class classification

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: accuracy
    :rtype: float
    """
    n = len(X)
    Y_pred_probs = model.predict(theta, X)
    return np.sum(Y_pred_probs[np.arange(n), Y]) / n


def confusion_matrix(model, theta, X, Y, l_i, l_k, **kwargs):
    """
    Get the probability of predicting class label l_k
    if the true class label was l_i. This is the C[l_i,l_k]
    element of the confusion matrix, C. Let:
            i = number of datapoints
            j = number of features (including bias term, if provied)
            k = number of classes

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: array of shape (j,k)
    :param X: The features
    :type X: array of shape (i,j)
    :param Y: The labels
    :type Y: array of shape (i,k)
    :param l_i: The index in the confusion matrix
            corresponding to the true label (row)
    :type l_i: int
    :param l_k: The index in the confusion matrix
            corresponding to the predicted label (column)
    :type l_k: int

    :return: The element
    :rtype: float
    """
    Y_pred = model.predict(theta, X)  # i x k
    true_mask = Y == l_i  # length i
    N_mask = sum(true_mask)

    res = sum(Y_pred[:, l_k][true_mask]) / N_mask
    return res


""" RL """


def IS_estimate(model, theta, episodes, **kwargs):
    """Calculate the unweighted importance sampling estimate
    on all episodes in the dataframe

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param episodes: List of episodes
    :return: The IS estimate calculated over all episodes
    :rtype: float
    """
    # Possible that weighted returns were calculated ahead of time.
    # If not, then calculated them here

    # Calculate the expected returns of the primary reward    
    if "gamma" in model.env_kwargs:
        gamma = model.env_kwargs["gamma"]
    else:
        gamma = 1.0
    weighted_returns = [
        weighted_sum_gamma(ep.rewards, gamma=gamma) for ep in episodes
    ]

    IS_estimate = 0
    for ii, ep in enumerate(episodes):
        pi_news = model.get_probs_from_observations_and_actions(
            theta, ep.observations, ep.actions, ep.action_probs
        )
        pi_ratios = pi_news / ep.action_probs
        pi_ratio_prod = np.prod(pi_ratios)

        IS_estimate += pi_ratio_prod * weighted_returns[ii]

    IS_estimate /= len(episodes)

    return IS_estimate

def PDIS_estimate(model, theta, episodes, **kwargs)->float:
    """Calculate per decision importance sampling estimate
    on all episodes in the dataframe

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param episodes: List of episodes
    :return: The PDIS estimate calculated over all episodes
    :rtype: float
    """

    gamma = model.env_kwargs["gamma"] if "gamma" in model.env_kwargs else 1.0
    PDIS_est = 0.
    for ep in episodes:
        discount = np.power(gamma, range(len(ep.rewards)))
        pi_news = model.get_probs_from_observations_and_actions(
            theta, ep.observations, ep.actions, ep.action_probs
        )
        pi_ratios = pi_news / ep.action_probs

        # autograd doesn't support np.cumprod
        pi_ratio_prods = custom_cumprod(pi_ratios)

        PDIS_est += np.sum(pi_ratio_prods * discount * ep.rewards)

    PDIS_est /= len(episodes)

    return PDIS_est

