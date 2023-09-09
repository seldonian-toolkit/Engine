""" Objective functions """

import autograd.numpy as np  # Thinly-wrapped version of Numpy
import math

from seldonian.utils.stats_utils import (weighted_sum_gamma,
    custom_cumprod, stability_const)

""" Convenience functions """

def batcher(func, N, batch_size, num_batches):
    """Calls function func num_batches times,
    batching up the inputs.

    :param func: The function you want to call
    :param N: The total number of datapoints
    :type N: int
    :param batch_size: The size of each batch
    :type batch_size: int
    :param num_batches: The number of batches
    :type num_batches: int

    :return: A wrapper function that does the actual function calls
    """
    def wrapper(*args, **kw):
        regime = kw["regime"]
        model = args[0]
        theta = args[1]
        if regime == "supervised_learning":
            features = args[2]
            labels = args[3]
        elif regime == "reinforcement_learning":
            episodes = args[2]
            weighted_returns = kw["weighted_returns"]
        if num_batches > 1:
            res = np.zeros(N)
            batch_start = 0
            for i in range(num_batches):
                batch_end = batch_start + batch_size
                if regime == "supervised_learning":
                    if type(features) == list:
                        features_batch = [x[batch_start:batch_end] for x in features]
                    else:
                        features_batch = features[batch_start:batch_end]

                    labels_batch = labels[batch_start:batch_end]
                    batch_args = [model, theta, features_batch, labels_batch]

                elif regime == "reinforcement_learning":
                    episodes_batch = episodes[batch_start:batch_end]
                    weighted_returns_batch = weighted_returns[batch_start:batch_end]
                    batch_args = [model, theta, episodes_batch, weighted_returns_batch]

                res[batch_start:batch_end] = func(*batch_args, **kw)

                batch_start = batch_end
        else:
            res = func(*args, **kw)
        return res

    return wrapper


def _setup_params_for_stat_funcs(model, theta, data_dict, sub_regime, **kwargs):
    """Set up the args,kwargs to pass to the 
    zhat functions.
    
    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param data_dict: Contains the features and labels
    :type data_dict: dict
    :param sub_regime: The specific type of ML problem, e.g. "classification"
    :type sub_regime: str

    :return: (args, msr_func_kwargs, num_datapoints), 
        a tuple consisting of positional arguments (args), 
        keyword arguments (msr_func_kwargs) and the total
        number of datapoints to consider (num_datapoints).
    """
    regime = kwargs["regime"]
    dataset = kwargs["dataset"]
    msr_func_kwargs = {"regime": regime}
    
    if regime == "supervised_learning":
        num_datapoints = len(data_dict["labels"])
        args = [model, theta, data_dict["features"], data_dict["labels"]]
        sub_regime = dataset.meta.sub_regime
        msr_func_kwargs["sub_regime"] = sub_regime
        if "class_index" in kwargs:
            msr_func_kwargs["class_index"] = kwargs["class_index"]
        if "cm_true_index" in kwargs:
            msr_func_kwargs["l_i"] = kwargs["cm_true_index"]
        if "cm_pred_index" in kwargs:
            msr_func_kwargs["l_k"] = kwargs["cm_pred_index"]

    elif regime == "reinforcement_learning":
        episodes = data_dict["episodes"]
        num_datapoints = len(episodes)
        msr_func_kwargs["weighted_returns"] = data_dict["weighted_returns"]
        args = [model, theta, episodes]

    return args,msr_func_kwargs,num_datapoints


def sample_from_statistic(model, statistic_name, theta, data_dict, **kwargs):
    """Calculate a statistical function for each observation
    in the sample.

    :param model: SeldonianModel instance
    :param statistic_name: The name of the statistic to evaluate
    :type statistic_name: str, e.g. 'FPR'
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param data_dict: Contains the features and labels
    :type data_dict: dict

    :return: The evaluated statistic for each observation in the sample
    :rtype: numpy ndarray(float)
    """
    branch = kwargs["branch"]
    regime = kwargs["regime"]
    sub_regime = kwargs["dataset"].meta.sub_regime
    
    (
        args,
        msr_func_kwargs,
        num_datapoints
    ) = _setup_params_for_stat_funcs(
        model=model,
        theta=theta, 
        data_dict=data_dict,
        sub_regime=sub_regime,
        **kwargs
    )

    msr_func = measure_function_vector_mapper[statistic_name]

    if branch == "candidate_selection":
        return msr_func(*args, **msr_func_kwargs)

    elif branch == "safety_test":
        if "batch_size_safety" in kwargs:
            if kwargs["batch_size_safety"] is None:
                batch_size_safety = num_datapoints
                num_batches = 1
            else:
                batch_size_safety = kwargs["batch_size_safety"]
                num_batches = math.ceil(num_datapoints / batch_size_safety)

        else:
            batch_size_safety = num_datapoints
            num_batches = 1
        return batcher(
            msr_func, N=num_datapoints, batch_size=batch_size_safety, num_batches=num_batches
        )(*args, **msr_func_kwargs)


def evaluate_statistic(model, statistic_name, theta, data_dict, **kwargs):
    """Evaluate the mean of a statistical function over the whole sample provided.

    :param model: SeldonianModel instance
    :param statistic_name: The name of the statistic to evaluate
    :type statistic_name: str, e.g. 'FPR' for false positive rate
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param data_dict: Contains the features and labels
    :type data_dict: dict

    :return: The evaluated statistic over the whole sample
    :rtype: float
    """
    branch = kwargs["branch"]
    regime = kwargs["regime"]
    sub_regime = kwargs["dataset"].meta.sub_regime
    
    (
        args,
        msr_func_kwargs,
        num_datapoints
    ) = _setup_params_for_stat_funcs(
        model=model,
        theta=theta, 
        data_dict=data_dict,
        sub_regime=sub_regime,
        **kwargs
    )

    msr_func = measure_function_vector_mapper[statistic_name]

    if branch == "candidate_selection":
        return np.mean(msr_func(*args, **msr_func_kwargs))

    elif branch == "safety_test":
        if "batch_size_safety" in kwargs:
            if kwargs["batch_size_safety"] is None:
                batch_size_safety = num_datapoints
                num_batches = 1
            else:
                batch_size_safety = kwargs["batch_size_safety"]
                num_batches = math.ceil(num_datapoints / batch_size_safety)

        else:
            batch_size_safety = num_datapoints
            num_batches = 1

        return np.mean(
            batcher(
                msr_func,
                N=num_datapoints,
                batch_size=batch_size_safety,
                num_batches=num_batches,
            )(*args, **msr_func_kwargs)
        )


""" Regression zhat functions """

def vector_Squared_Error(model, theta, X, Y, **kwargs):
    """Calculate squared error for each observation
    in the dataset

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: vector of mean squared error values
    :rtype: numpy ndarray(float)
    """
    prediction = model.predict(theta, X)
    return pow(prediction - Y, 2)


def vector_Error(model, theta, X, Y, **kwargs):
    """Calculate error (Y_hat - Y) for each observation
    in the dataset

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: vector of error values
    :rtype: numpy ndarray(float)
    """
    prediction = model.predict(theta, X)
    return prediction - Y


""" Classification zhat functions """


def vector_Positive_Rate(model, theta, X, Y, **kwargs):
    """
    Calculate positive rate
    for each observation. Meaning depends on whether
    binary or multi-class classification.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Positive rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    if "class_index" in kwargs:
        return _vector_Positive_Rate_multiclass(
            model, theta, X, Y, class_index=kwargs["class_index"]
        )
    else:
        return _vector_Positive_Rate_binary(model, theta, X, Y)


def _vector_Positive_Rate_binary(model, theta, X, Y, **kwargs):
    """
    Calculate positive rate
    for each observation. This is the probability of 
    predicting the positive class.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Positive rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    prediction = model.predict(theta, X)
    return prediction


def _vector_Positive_Rate_multiclass(model, theta, X, Y, class_index, **kwargs):
    """
    Calculate positive rate
    for each observation. This is the probability of 
    predicting class=class_index.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray
    :param class_index: The index of the class label
    :type class_index: int, 0-indexed

    :return: Positive rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    prediction = model.predict(theta, X)
    return prediction[:, class_index]


def vector_Negative_Rate(model, theta, X, Y, **kwargs):
    """
    Calculate negative rate
    for each observation. Meaning depends on whether
    binary or multi-class classification.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Negative rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    if "class_index" in kwargs:
        return _vector_Negative_Rate_multiclass(
            model, theta, X, Y, class_index=kwargs["class_index"]
        )
    else:
        return _vector_Negative_Rate_binary(model, theta, X, Y)


def _vector_Negative_Rate_binary(model, theta, X, Y, **kwargs):
    """
    Calculate negative rate
    for each observation. This is the probability
    of predicting the negative class.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Negative rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    prediction = model.predict(theta, X)
    return 1.0 - prediction


def _vector_Negative_Rate_multiclass(model, theta, X, Y, class_index, **kwargs):
    """
    Calculate negative rate
    for each observation. This is the probability
    of predicting a class other than class_index.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray
    :param class_index: The index of the class label
    :type class_index: int, 0-indexed

    :return: Negative rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    prediction = model.predict(theta, X)
    return 1.0 - prediction[:, class_index]


def vector_False_Positive_Rate(model, theta, X, Y, **kwargs):
    """
    Calculate false positive rate
    for each observation. Meaning depends on whether
    binary or multi-class classification.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: False positive rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    if "class_index" in kwargs:
        return _vector_False_Positive_Rate_multiclass(
            model, theta, X, Y, class_index=kwargs["class_index"]
        )
    else:
        return _vector_False_Positive_Rate_binary(model, theta, X, Y)


def _vector_False_Positive_Rate_binary(model, theta, X, Y, **kwargs):
    """
    Calculate false positive rate
    for each observation. This is the probability of predicting
    the positive class when the true label is the negative class.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: False positive rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    prediction = model.predict(theta, X)
    neg_mask = Y != 1.0  # this includes false positives and true negatives
    return prediction[neg_mask]


def _vector_False_Positive_Rate_multiclass(model, theta, X, Y, class_index, **kwargs):
    """
    Calculate false positive rate
    for each observation. This is the probability of predicting
    the class=class_index when the true label is any other class.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray
    :param class_index: The index of the class label
    :type class_index: int, 0-indexed

    :return: False positive rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    prediction = model.predict(theta, X)
    other_mask = Y != class_index
    return prediction[:, class_index][other_mask]


def vector_False_Negative_Rate(model, theta, X, Y, **kwargs):
    """
    Calculate false negative rate
    for each observation. Meaning depends on whether
    binary or multi-class classification.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: False negative rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    if "class_index" in kwargs:
        return _vector_False_Negative_Rate_multiclass(
            model, theta, X, Y, class_index=kwargs["class_index"]
        )
    else:
        return _vector_False_Negative_Rate_binary(model, theta, X, Y)


def _vector_False_Negative_Rate_binary(model, theta, X, Y, **kwargs):
    """
    Calculate false negative rate
    for each observation. This is the probability of predicting 
    the negative class when the true label was the positive class. 

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: False negative rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    prediction = model.predict(theta, X)
    pos_mask = Y == 1.0  # this includes false positives and true negatives
    return 1.0 - prediction[pos_mask]


def _vector_False_Negative_Rate_multiclass(model, theta, X, Y, class_index, **kwargs):
    """
    Calculate false negative rate
    for each observation. This is the probability of predicting being
    in any class besides class_index when the true label is class_index

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray
    :param class_index: The index of the class label
    :type class_index: int, 0-indexed

    :return: False negative rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    prediction = model.predict(theta, X)
    pos_mask = Y == class_index  # this includes false positives and true negatives
    return (1.0 - prediction[:, class_index])[pos_mask]


def vector_True_Positive_Rate(model, theta, X, Y, **kwargs):
    """
    Calculate true positive rate
    for each observation. Meaning depends on whether
    binary or multi-class classification.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: True positive rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    if "class_index" in kwargs:
        return _vector_True_Positive_Rate_multiclass(
            model, theta, X, Y, class_index=kwargs["class_index"]
        )
    else:
        return _vector_True_Positive_Rate_binary(model, theta, X, Y)


def _vector_True_Positive_Rate_binary(model, theta, X, Y, **kwargs):
    """
    Calculate true positive rate
    for each observation. This is the probability of predicting the 
    positive class when the true label is the positive class. 

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: True positive rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    prediction = model.predict(theta, X)
    pos_mask = Y == 1.0  # this includes false positives and true negatives
    return prediction[pos_mask]


def _vector_True_Positive_Rate_multiclass(model, theta, X, Y, class_index, **kwargs):
    """
    Calculate true positive rate
    for each observation. This is the probability of predicting  
    class=class_index when the true label is class_index.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray
    :param class_index: The index of the class label
    :type class_index: int, 0-indexed

    :return: True positive rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    prediction = model.predict(theta, X)
    pos_mask = Y == class_index  # this includes false positives and true negatives
    return (prediction[:, class_index])[pos_mask]


def vector_True_Negative_Rate(model, theta, X, Y, **kwargs):
    """
    Calculate true negative rate
    for each observation. Meaning depends on whether
    binary or multi-class classification.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: True negative rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    if "class_index" in kwargs:
        return _vector_True_Negative_Rate_multiclass(
            model, theta, X, Y, class_index=kwargs["class_index"]
        )
    else:
        return _vector_True_Negative_Rate_binary(model, theta, X, Y)


def _vector_True_Negative_Rate_binary(model, theta, X, Y, **kwargs):
    """
    Calculate true negative rate
    for each observation. This is the probability of predicting
    the negative class when the true label was the negative class.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: True negative rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    prediction = model.predict(theta, X)
    neg_mask = Y != 1.0
    return 1.0 - prediction[neg_mask]


def _vector_True_Negative_Rate_multiclass(model, theta, X, Y, class_index, **kwargs):
    """
    Calculate true negative rate
    for each observation. This is the probability 
    of predicting class!=class_index when the true label was not class_index.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray
    :param class_index: The index of the class label
    :type class_index: int, 0-indexed

    :return: True negative rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    prediction = model.predict(theta, X)
    neg_mask = Y != class_index
    return (1.0 - prediction[:, class_index])[neg_mask]


def vector_Error_Rate(model, theta, X, Y, **kwargs):
    """
    Calculate error rate for each observation.
    This is 1 - the probability of predicting the
    correct label.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Error rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    if kwargs["sub_regime"] == "multiclass_classification":
        return _vector_Error_Rate_multiclass(model, theta, X, Y, **kwargs)
    else:
        return _vector_Error_Rate_binary(model, theta, X, Y, **kwargs)


def _vector_Error_Rate_binary(model, theta, X, Y, **kwargs):
    """
    Calculate error rate for each observation.
    This is 1 - the probability of predicting the
    correct label.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Error rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    Y_pred_probs = model.predict(theta, X)
    # Get probabilities of true positives and true negatives
    # Use the vector Y_pred as it already has the true positive
    # probs. Just need to replace the probabilites in the neg mask with 1-prob
    return Y*(1-Y_pred_probs) + (1-Y)*Y_pred_probs


def _vector_Error_Rate_multiclass(model, theta, X, Y, **kwargs):
    """
    Calculate error rate for each observation.
    This is 1 - the probability of predicting the
    correct label.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Error rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    n = len(X)
    Y_pred_probs = model.predict(theta, X)
    return 1.0 - Y_pred_probs[np.arange(n), Y]


def vector_Accuracy(model, theta, X, Y, **kwargs):
    """
    Calculate probabilistic accuracy for each observation.
    This is the probability of predicting the
    correct label, and equivalent to 1 - error rate.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Error rate for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    if kwargs["sub_regime"] == "multiclass_classification":
        return _vector_Accuracy_multiclass(model, theta, X, Y, **kwargs)
    else:
        return _vector_Accuracy_binary(model, theta, X, Y, **kwargs)


def _vector_Accuracy_binary(model, theta, X, Y, **kwargs):
    """
    Calculate probabilistic accuracy for each observation.
    This is the probability of predicting the
    correct label, and equivalent to 1 - error rate.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Accuracy for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    Y_pred_probs = model.predict(theta, X)
    return Y*Y_pred_probs + (1-Y)*(1-Y_pred_probs)


def _vector_Accuracy_multiclass(model, theta, X, Y, **kwargs):
    """
    Calculate probabilistic accuracy for each observation.
    This is the probability of predicting the
    correct label, and equivalent to 1 - error rate.

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param X: The features
    :type X: numpy ndarray
    :param Y: The labels
    :type Y: numpy ndarray

    :return: Accuracy for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    n = len(X)
    Y_pred_probs = model.predict(theta, X)
    return Y_pred_probs[np.arange(n), Y]


def vector_confusion_matrix(model, theta, X, Y, l_i, l_k, **kwargs):
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

    :return: C[l_i,l_k] for each observation
    :rtype: numpy ndarray(float between 0 and 1)
    """
    Y_pred = model.predict(theta, X)  # i x k
    true_mask = Y == l_i  # length i

    N_mask = sum(true_mask)
    res = Y_pred[:, l_k][true_mask]
    return res


""" RL zhat functions """

def vector_IS_estimate(model, theta, episodes, weighted_returns, **kwargs):
    """Calculate the unweighted importance sampling estimate
    on each episodes in the dataframe

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param episodes: List of episodes
    :param weighted_returns: A pre-calculated list of weighted returns
        from the reward that is present in the constraint

    :return: A vector of IS estimates calculated for each episode
    :rtype: numpy ndarray(float)
    """

    result = []
    for ii, ep in enumerate(episodes):
        pi_news = model.get_probs_from_observations_and_actions(
            theta, ep.observations, ep.actions, ep.action_probs
        )
        pi_ratio_prod = np.prod(pi_news / ep.action_probs)
        result.append(pi_ratio_prod * weighted_returns[ii])

    return np.array(result)


def vector_PDIS_estimate(model, theta, episodes, weighted_returns, **kwargs):
    """Calculate per decision importance sampling estimate
    on each episodes in the dataframe

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param episodes: List of episodes
    :param weighted_returns: A pre-calculated list of weighted returns
        from the reward that is present in the constraint

    :return: A vector of PDIS estimates calculated for each episode
    :rtype: numpy ndarray(float)
    """

    gamma = model.env_kwargs["gamma"] if "gamma" in model.env_kwargs else 1.0
    PDIS_vector = []
    for ep in episodes:
        discount = np.power(gamma, range(len(ep.rewards)))
        pi_news = model.get_probs_from_observations_and_actions(
            theta, ep.observations, ep.actions, ep.action_probs
        )
        pi_ratios = pi_news / ep.action_probs
        
        # autograd doesn't support np.cumprod
        pi_ratio_prods = custom_cumprod(pi_ratios)

        PDIS_vector.append( np.sum(pi_ratio_prods * discount * ep.rewards) )

    return np.array(PDIS_vector)


def vector_WIS_estimate(model, theta, episodes, weighted_returns, **kwargs):
    """Calculate weighted importance sampling estimate
    on each episodes in the dataframe

    :param model: SeldonianModel instance
    :param theta: The parameter weights
    :type theta: numpy ndarray
    :param episodes: List of episodes
    :param weighted_returns: A pre-calculated list of weighted returns
        from the reward that is present in the constraint

    :return: A vector of WIS estimates calculated for each episode, 
        such that the mean of this vector will be the WIS estimate.
    :rtype: numpy ndarray(float)
    """
    gamma = model.env_kwargs["gamma"] if "gamma" in model.env_kwargs else 1.0
    n = len(episodes)
    rho_array = []
    for ii, ep in enumerate(episodes):
        # Get pi_new for each timestep in this ep
        pi_news = model.get_probs_from_observations_and_actions(
            theta, ep.observations, ep.actions, ep.action_probs
        )
        rho_array.append(np.prod(pi_news/ep.action_probs))
    rho_array = np.array(rho_array)
    WIS_vector = n*rho_array*weighted_returns/np.sum(rho_array)
    return WIS_vector


def vector_auxiliary_return_US_estimate(model, theta, episodes, weighted_returns, **kwargs):
    """Get the auxiliary reward returns for episodes
     whose actions fall within the theta bounding box.
     This function is used for constraints, unlike Bounding_box_estimate()
     which is used for primary objective functions.
    """
    crmin,crmax,cfmin,cfmax = model.policy.theta2crcf(theta)

    returns_inside_theta_box = []
    for ii, ep in enumerate(episodes):
        cr,cf = ep.actions[0] # behavior policy action
        secondary_return = ep.alt_rewards[0][0]
        # theta is crmin, crmax, cfmin, cfmax
        if (crmin <= cr <= crmax) and (cfmin <= cf <= cfmax):
            returns_inside_theta_box.append(secondary_return)
    n_inside_box = len(returns_inside_theta_box)
    return np.array(returns_inside_theta_box)


""" Measure function mapper that maps from string that appears in the constraint string to the appropriate function. """

measure_function_vector_mapper = {
    "Mean_Squared_Error": vector_Squared_Error,
    "Mean_Error": vector_Error,
    "PR": vector_Positive_Rate,
    "NR": vector_Negative_Rate,
    "FPR": vector_False_Positive_Rate,
    "FNR": vector_False_Negative_Rate,
    "TPR": vector_True_Positive_Rate,
    "TNR": vector_True_Negative_Rate,
    "ACC": vector_Accuracy,
    "J_pi_new": vector_IS_estimate,
    'J_pi_new_PDIS':vector_PDIS_estimate,
    'J_pi_new_WIS':vector_WIS_estimate,
    'J_pi_new_US':vector_auxiliary_return_US_estimate,
}