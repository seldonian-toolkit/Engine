import os
import pytest
import autograd.numpy as np

from seldonian.spec import SupervisedSpec
from seldonian.models import objectives
from seldonian.models.models import *
from seldonian.models.trees.sktree_model import SeldonianDecisionTree
from seldonian.models.trees.skrandomforest_model import SeldonianRandomForest
from seldonian.seldonian_algorithm import SeldonianAlgorithm


def test_linear_regression_model():
    model = LinearRegressionModel()
    assert model.has_intercept == True
    X = np.array(
        [[0.0, 0.0], [0.25, 0.5], [0.5, 1.0], [0.75, 1.5]]
    )  # i samples x j features
    Y = np.array([0.0, -0.5, 0.5, 1.0])  # length i, true labels
    theta_init = np.array([0.0, -1.0, 1.0])  # j+1 in length to account for intercept
    y_pred = model.predict(theta_init, X)
    answer1 = [0.0, 0.25, 0.5, 0.75]
    assert np.allclose(y_pred, answer1)
    theta_fitted = model.fit(X, Y)
    answer2 = [-0.35, 0.32, 0.64]
    assert np.allclose(theta_fitted, answer2)


def test_binary_logistic_regression_model():
    model = BinaryLogisticRegressionModel()
    assert model.has_intercept == True
    X = np.array(
        [[0.0, 0.0], [0.25, 0.5], [0.5, 1.0], [0.75, 1.5]]
    )  # i samples x j features
    Y = np.array([0, 0, 1, 1])  # length i, true labels
    theta_init = np.array([0.0, -1.0, 1.0])  # j+1 in length to account for intercept
    y_pred = model.predict(theta_init, X)
    # print(y_pred)
    answer1 = [0.5, 0.5621765, 0.62245933, 0.6791787]
    assert np.allclose(y_pred, answer1)
    theta_fitted = model.fit(X, Y)
    print(theta_fitted)
    answer2 = [-0.68058112, 0.36297811, 0.72595622]
    assert np.allclose(theta_fitted, answer2)


def test_multiclass_logistic_regression_model():
    model = MultiClassLogisticRegressionModel()
    assert model.has_intercept == True
    # i=4 samples, j=2 features, k=3 classes.
    X = np.array([[0.0, 0.0], [0.25, 0.5], [0.5, 1.0], [0.75, 1.5]])  # i x j
    Y = np.array([0, 1, 2, 0])  # length i, true labels
    theta_init = np.array(
        [
            [0.0, -1.0, 1.0],
            [-1.0, -0.5, 0.5],
            [1.0, 0.5, -0.5],
        ]
    )  # j+1 x k classes in length to account for intercept
    y_pred = model.predict(theta_init, X)

    # print(y_pred)
    answer1 = [
        [0.24472847, 0.09003057, 0.66524096],
        [0.31319506, 0.10167955, 0.58512540],
        [0.38902480, 0.11145747, 0.49951773],
        [0.46831026, 0.11840738, 0.41328236],
    ]
    assert np.allclose(y_pred, answer1)
    theta_fitted = model.fit(X, Y)
    answer2 = [
        [4.67337353e-01, -6.48501834e-02, -4.02487170e-01],
        [-5.58199249e-06, -9.00361538e-02, 9.00417358e-02],
        [-1.11639850e-05, -1.80072308e-01, 1.80083472e-01],
    ]
    assert np.allclose(theta_fitted, answer2)


def test_sklearn_dtree():
    model = SeldonianDecisionTree(max_depth=4, criterion="entropy")
    assert model.has_intercept == False
    X = np.random.uniform(-1, 1, (100, 4))  # i samples x j features
    Y = np.random.randint(0, 2, 100)  # length i, true labels
    theta_init = model.fit(X, Y)
    y_pred = model.predict(theta_init, X)
    assert len(y_pred) == 100
    prob_mask = np.logical_and(y_pred >= 0, y_pred <= 1)
    assert all(prob_mask)


def test_sklearn_random_forest():
    model = SeldonianRandomForest(max_depth=4, n_estimators=10)
    assert model.has_intercept == False
    X = np.random.uniform(-1, 1, (100, 4))  # i samples x j features
    Y = np.random.randint(0, 2, 100)  # length i, true labels
    theta_init = model.fit(X, Y)
    y_pred = model.predict(theta_init, X)
    assert len(y_pred) == 100
    prob_mask = np.logical_and(y_pred >= 0, y_pred <= 1)
    assert all(prob_mask)
