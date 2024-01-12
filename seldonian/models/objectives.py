""" Objective functions """

import autograd.numpy as np  # Thinly-wrapped version of Numpy
import math

from seldonian.utils.stats_utils import (
    weighted_sum_gamma,
    custom_cumprod,
    stability_const,
)
from seldonian.models.models import BaseLogisticRegressionModel

""" Supervised learning objectives """

####### Regression ########

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

    :return: vector gradient d(MSE)/dtheta
    :rtype: numpy ndarray
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

    :return: Sample mean error
    :rtype: float
    """
    n = len(X)
    prediction = model.predict(theta, X)  # vector of values
    res = sum(prediction - Y) / n
    return res

def gradient_Bounded_Squared_Error(model, theta, X, Y, **kwargs):
    """Analytical gradient of the bounded squared error (BSE)

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: d(BSE)/dtheta
    :rtype: numpy ndarray
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

####### Classification ########

def binary_logistic_loss(model, theta, X, Y, **kwargs):
    """Calculate mean logistic loss
    over all data points for binary classification

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: mean logistic loss
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
    """Gradient of binary logistic loss w.r.t. theta.
    WARNING: This is only valid for binary logistic regression models!
    DO NOT USE FOR NEURAL NETWORKS.
    Also, the number of parameters must be the same as the number of model weights.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: d(log loss)/dtheta
    :rtype: float
    """
    assert isinstance(model, BaseLogisticRegressionModel)

    h = model.predict(theta, X)
    X_withintercept = np.hstack([np.ones((len(X), 1)), np.array(X)])
    res = (1 / len(X)) * np.dot(X_withintercept.T, (h - Y))
    return res

def multiclass_logistic_loss(model, theta, X, Y, **kwargs):
    """Calculate mean logistic loss
    over all data points for multi-class classification

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: mean logistic loss
    :rtype: float
    """
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
    Calculate mean positive rate
    for the whole sample. This has slightly different meanings
    depending on whether we're in binary or multi-class setting.

    :param model: SeldonianModel instance:param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Mean positive rate
    :rtype: float between 0 and 1
    """
    if "class_index" in kwargs:
        return _Positive_Rate_multiclass(
            model, theta, X, Y, class_index=kwargs["class_index"]
        )
    else:
        return _Positive_Rate_binary(model, theta, X, Y)

def _Positive_Rate_binary(model, theta, X, Y, **kwargs):
    """
    Calculate mean positive rate
    for the whole sample in the binary classification setting.
    This is the mean probability of predicting the positive class.

    :param model: SeldonianModel instance:param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Mean positive rate
    :rtype: float between 0 and 1
    """
    prediction = model.predict(theta, X)
    return np.sum(prediction) / len(X)  # if all 1s then PR=1.

def _Positive_Rate_multiclass(model, theta, X, Y, class_index, **kwargs):
    """
    Calculate mean positive rate
    for the whole sample in the multi-class classification setting.
    This is the mean probability of predicting class=class_index.

    :param model: SeldonianModel instance:param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray
    :param class_index: The index of the class label
    :type class_index: int, 0-indexed

    :return: Mean positive rate
    :rtype: float between 0 and 1
    """
    prediction = model.predict(theta, X)
    return np.sum(prediction[:, class_index]) / len(X)  # if all 1s then PR=1.

def Negative_Rate(model, theta, X, Y, **kwargs):
    """
    Calculate mean negative rate
    for the whole sample.
    This has slightly different meanings
    depending on whether we're in binary or multi-class setting.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Mean negative rate
    :rtype: float between 0 and 1
    """
    if "class_index" in kwargs:
        return _Negative_Rate_multiclass(
            model, theta, X, Y, class_index=kwargs["class_index"]
        )
    else:
        return _Negative_Rate_binary(model, theta, X, Y)

def _Negative_Rate_binary(model, theta, X, Y, **kwargs):
    """
    Calculate mean negative rate
    for the whole sample.
    This is the mean probability of predicting the negative class.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Mean negative rate
    :rtype: float between 0 and 1
    """
    prediction = model.predict(theta, X)
    return np.sum(1.0 - prediction) / len(X)  # if all 1s then PR=1.

def _Negative_Rate_multiclass(model, theta, X, Y, class_index, **kwargs):
    """
    Calculate mean negative rate
    for the whole sample.
    This is the mean probability of predicting any class except class_index.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray
    :param class_index: The index of the class label
    :type class_index: int, 0-indexed

    :return: Mean negative rate
    :rtype: float between 0 and 1
    """
    prediction = model.predict(theta, X)
    return np.sum(1.0 - prediction[:, class_index]) / len(X)

def False_Positive_Rate(model, theta, X, Y, **kwargs):
    """
    Calculate probabilistic mean false positive rate
    over the whole sample. This has slightly different meanings
    depending on binary vs. multi-class setting.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: mean false positive rate
    :rtype: float between 0 and 1
    """
    if "class_index" in kwargs:
        return _False_Positive_Rate_multiclass(
            model, theta, X, Y, class_index=kwargs["class_index"]
        )
    else:
        return _False_Positive_Rate_binary(model, theta, X, Y)

def _False_Positive_Rate_binary(model, theta, X, Y, **kwargs):
    """
    Calculate mean false positive rate
    over the whole sample. This is the mean probability
    of predicting the positive class when the true label
    is the negative class.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: mean false positive rate
    :rtype: float between 0 and 1
    """

    prediction = model.predict(theta, X)
    neg_mask = Y != 1.0
    return np.sum(prediction[neg_mask]) / len(X[neg_mask])

def _False_Positive_Rate_multiclass(model, theta, X, Y, class_index, **kwargs):
    """
    Calculate mean false positive rate
    over the whole sample. This is the mean probability of predicting
    class=class_index, subject to the true label being any other class.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray
    :param class_index: The index of the class label
    :type class_index: int, 0-indexed

    :return: mean false positive rate
    :rtype: float between 0 and 1
    """

    prediction = model.predict(theta, X)

    neg_mask = Y != class_index
    return np.sum(prediction[:, class_index][neg_mask]) / len(X[neg_mask])

def False_Negative_Rate(model, theta, X, Y, **kwargs):
    """
    Calculate probabilistic mean false negative rate
    over the whole sample. This has slightly different meanings
    depending on binary vs. multi-class setting.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Mean false negative rate
    :rtype: float between 0 and 1
    """
    if "class_index" in kwargs:
        return _False_Negative_Rate_multiclass(
            model, theta, X, Y, class_index=kwargs["class_index"]
        )
    else:
        return _False_Negative_Rate_binary(model, theta, X, Y)

def _False_Negative_Rate_binary(model, theta, X, Y, **kwargs):
    """
    Calculate probabilistic mean false negative rate
    over the whole sample. This is the mean probability
    of predicting the negative class subject to the truth being the positive class

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Mean false negative rate
    :rtype: float between 0 and 1
    """

    prediction = model.predict(theta, X)
    pos_mask = Y == 1.0
    return np.sum(1.0 - prediction[pos_mask]) / len(X[pos_mask])

def _False_Negative_Rate_multiclass(model, theta, X, Y, class_index, **kwargs):
    """
    Calculate probabilistic mean false negative rate
    over the whole sample. This is the mean probability
    of predicting a class other than class_index, subject to the
    true label being class_index.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray
    :param class_index: The index of the class label
    :type class_index: int, 0-indexed

    :return: Mean false negative rate
    :rtype: float between 0 and 1
    """
    prediction = model.predict(theta, X)
    pos_mask = Y == class_index
    return np.sum(1.0 - prediction[:, class_index][pos_mask]) / len(X[pos_mask])

def True_Positive_Rate(model, theta, X, Y, **kwargs):
    """
    Calculate mean true positive rate
    for the whole sample. This has slightly different meanings
    depending on binary vs. multi-class setting.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Mean true positive rate
    :rtype: float between 0 and 1
    """
    if "class_index" in kwargs:
        return _True_Positive_Rate_multiclass(
            model, theta, X, Y, class_index=kwargs["class_index"]
        )
    else:
        return _True_Positive_Rate_binary(model, theta, X, Y)

def _True_Positive_Rate_binary(model, theta, X, Y, **kwargs):
    """
    Calculate mean true positive rate
    for the whole sample. This is the mean probability
    of predicting the positive class when the true label is the positive class

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Mean true positive rate
    :rtype: float between 0 and 1
    """

    prediction = model.predict(theta, X)
    pos_mask = Y == 1.0
    return np.sum(prediction[pos_mask]) / len(X[pos_mask])

def _True_Positive_Rate_multiclass(model, theta, X, Y, class_index, **kwargs):
    """
    Calculate mean true positive rate
    for the whole sample. This is the mean probability
    of predicting class=class_index when the true label is class_index

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray
    :param class_index: The index of the class label
    :type class_index: int, 0-indexed

    :return: Mean true positive rate
    :rtype: float between 0 and 1
    """
    prediction = model.predict(theta, X)
    pos_mask = Y == class_index
    return np.sum(prediction[:, class_index][pos_mask]) / len(X[pos_mask])

def True_Negative_Rate(model, theta, X, Y, **kwargs):
    """
    Calculate mean true negative rate
    for the whole sample. This has slightly different meanings
    depending on binary vs. multi-class setting.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Mean true negative rate
    :rtype: float between 0 and 1
    """
    if "class_index" in kwargs:
        return _True_Negative_Rate_multiclass(
            model, theta, X, Y, class_index=kwargs["class_index"]
        )
    else:
        return _True_Negative_Rate_binary(model, theta, X, Y)

def _True_Negative_Rate_binary(model, theta, X, Y, **kwargs):
    """
    Calculate mean true negative rate
    for the whole sample. This is the mean probability
    of predicting the negative class when the true label was the negative class.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Mean true negative rate
    :rtype: float between 0 and 1
    """
    prediction = model.predict(theta, X)
    neg_mask = Y != 1.0
    return np.sum(1.0 - prediction[neg_mask]) / len(X[neg_mask])

def _True_Negative_Rate_multiclass(model, theta, X, Y, class_index, **kwargs):
    """
    Calculate mean true negative rate
    for the whole sample. This is the mean probability
    of predicting class!=class_index when the true label was not class_index

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray
    :param class_index: The index of the class label
    :type class_index: int, 0-indexed

    :return: Mean true negative rate
    :rtype: float between 0 and 1
    """
    prediction = model.predict(theta, X)
    neg_mask = Y != class_index
    return np.sum(1.0 - prediction[:, class_index][neg_mask]) / len(X[neg_mask])

def Error_Rate(model, theta, X, Y, **kwargs):
    """
    Calculate mean error rate over the whole sample.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray

    :return: Mean error rate 
    :rtype: float between 0 and 1
    """
    if kwargs["sub_regime"] == "multiclass_classification":
        return _Error_Rate_multiclass(model, theta, X, Y, **kwargs)
    else:
        return _Error_Rate_binary(model, theta, X, Y, **kwargs)

def _Error_Rate_binary(model, theta, X, Y, **kwargs):
    """Calculate mean error rate
    over all data points for binary classification

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: mean error rate between 0 and 1.
    :rtype: float
    """
    n = len(X)
    Y_pred_probs = model.predict(theta, X)
    res = np.sum(Y * (1 - Y_pred_probs) + (1 - Y) * Y_pred_probs) / n
    return res

def _Error_Rate_multiclass(model, theta, X, Y, **kwargs):
    """Calculate mean error rate
    for the whole sample.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: mean error rate
    :rtype: float between 0 and 1
    """
    n = len(X)
    Y_pred_probs = model.predict(theta, X)
    res = np.sum(1.0 - Y_pred_probs[np.arange(n), Y]) / n
    return res

def confusion_matrix(model, theta, X, Y, l_i, l_k, **kwargs):
    """Get the mean probability of predicting class label l_k
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

    :return: C[l_i,l_k]
    :rtype: float
    """
    Y_pred = model.predict(theta, X)  # i x k
    true_mask = Y == l_i  # length i
    N_mask = sum(true_mask)

    res = sum(Y_pred[:, l_k][true_mask]) / N_mask
    return res

""" Reinforcement learning objectives """

def IS_estimate(model, theta, episodes, **kwargs):
    """Calculate the vanilla importance sampling estimate
    using all episodes.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param episodes: List of episodes
    :return: The IS estimate 
    :rtype: float
    """

    if "gamma" in model.env_kwargs:
        gamma = model.env_kwargs["gamma"]
    else:
        gamma = 1.0
    # Calculate the expected returns of the primary reward under the behavior policy
    weighted_returns = [weighted_sum_gamma(ep.rewards, gamma=gamma) for ep in episodes]

    IS_est = 0
    for ii, ep in enumerate(episodes):
        pi_news = model.get_probs_from_observations_and_actions(
            theta, ep.observations, ep.actions, ep.action_probs
        )
        pi_ratios = pi_news / ep.action_probs
        pi_ratio_prod = np.prod(pi_ratios)

        IS_est += pi_ratio_prod * weighted_returns[ii]

    IS_est /= len(episodes)

    return IS_est

def PDIS_estimate(model, theta, episodes, **kwargs):
    """Calculate per-decision importance sampling (PDIS) estimate
    using all episodes.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param episodes: List of episodes
    :return: The PDIS estimate 
    :rtype: float
    """

    gamma = model.env_kwargs["gamma"] if "gamma" in model.env_kwargs else 1.0
    PDIS_est = 0.0
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

def WIS_estimate(model, theta, episodes, **kwargs):
    """Calculate the weighted importance sampling (WIS) estimate
    using all episodes. This is: sum(i=0 to n) { rho_i/rhosum} * G_i,
    where rhosum is sum(j=0 to n) {rho_j} and G_i is the discounted expected primary return.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param episodes: List of episodes
    :return: The WIS estimate 
    :rtype: float
    """

    if "gamma" in model.env_kwargs:
        gamma = model.env_kwargs["gamma"]
    else:
        gamma = 1.0
    # Calculate the expected returns of the primary reward under the behavior policy
    weighted_returns = np.array(
        [weighted_sum_gamma(ep.rewards, gamma=gamma) for ep in episodes]
    )
    # Calculate array of rho_j, which are the episode-wise importance weight products

    n = len(episodes)
    rho_array = []
    for ii, ep in enumerate(episodes):
        # Get pi_new for each timestep in this ep
        pi_news = model.get_probs_from_observations_and_actions(
            theta, ep.observations, ep.actions, ep.action_probs
        )
        rho_array.append(np.prod(pi_news / ep.action_probs))
    rho_array = np.array(rho_array)
    WIS_est = np.sum(rho_array * weighted_returns) / np.sum(rho_array)
    return WIS_est

def US_estimate(model, theta, episodes, **kwargs):
    """Get the expected return of the PRIMARY reward
    for behavior episodes whose actions (cr,cf)
    fall within the theta bounding box. 
    This function is the objective used in this 
    example: https://seldonian.cs.umass.edu/Tutorials/examples/diabetes/
    It is hardcoded for that specific application
    and in its current form should not be used
    for other problems. See https://arxiv.org/abs/1611.03451
    for the general form.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param episodes: List of episodes
    :return: The US estimate
    :rtype: float
    """

    crmin, crmax, cfmin, cfmax = model.policy.theta2crcf(theta)
    returns_inside_theta_box = []
    for ii, ep in enumerate(episodes):
        cr_b, cf_b = ep.actions[0]  # behavior policy action
        primary_return = ep.rewards[0]  # one reward per episode, so reward=return
        if (crmin <= cr_b <= crmax) and (cfmin <= cf_b <= cfmax):
            returns_inside_theta_box.append(primary_return)
    f = np.mean(returns_inside_theta_box)
    return f
