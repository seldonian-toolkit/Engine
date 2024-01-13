import os
import pytest
import autograd.numpy as np

from seldonian.spec import SupervisedSpec
from seldonian.models import objectives
from seldonian.parse_tree import zhat_funcs
from seldonian.models.models import *
from seldonian.seldonian_algorithm import SeldonianAlgorithm


def test_binary_classification_measure_functions():
    # i = 4 datapoints
    # j = 2 features
    # labels are 0 or 1
    model = BinaryLogisticRegressionModel()
    sub_regime = "binary_classification"
    Y = np.array([0, 0, 1, 1])  # length i, true labels
    X = np.array([[0.0, 0.0], [0.25, 0.5], [0.5, 1.0], [0.75, 1.5]])  # i x j
    theta = np.array([0.0, -1.0, 1.0])  # j+1 in length to account for intercept
    y_pred = model.predict(theta, X)  # length i
    # Avg statistics
    PR = objectives.Positive_Rate(model, theta, X, Y)
    assert PR == pytest.approx(0.5909536328157614)
    NR = objectives.Negative_Rate(model, theta, X, Y)
    assert NR == pytest.approx(1.0 - PR)
    FPR = objectives.False_Positive_Rate(model, theta, X, Y)
    # True label=0 was in first two datapoints. Avg(prob[1:3]) =~ 0.53
    assert FPR == pytest.approx(0.5310882504428991)
    FNR = objectives.False_Negative_Rate(model, theta, X, Y)
    # True label=1 was in last two datapoints. Avg(1.0-prob[2:]) =~ 0.35
    assert FNR == pytest.approx(0.3491809848113762)
    TPR = objectives.True_Positive_Rate(model, theta, X, Y)
    assert TPR == pytest.approx(1.0 - FNR)
    TNR = objectives.True_Negative_Rate(model, theta, X, Y)
    assert TNR == pytest.approx(1.0 - FPR)
    ERR = objectives.Error_Rate(model, theta, X, Y, sub_regime=sub_regime)
    assert ERR == pytest.approx(1.0 - 0.5598653825)
    # Vector statistics
    vector_PR = zhat_funcs.vector_Positive_Rate(model, theta, X, Y)
    assert np.allclose(vector_PR, y_pred)
    vector_NR = zhat_funcs.vector_Negative_Rate(model, theta, X, Y)
    assert np.allclose(vector_NR, 1.0 - y_pred)
    vector_FPR = zhat_funcs.vector_False_Positive_Rate(model, theta, X, Y)
    # True label=0 was in first two datapoints. prob[1:3] = [0.5,0.5621765]
    arcomp_FPR = y_pred[0:2]
    assert np.allclose(vector_FPR, arcomp_FPR)
    vector_FNR = zhat_funcs.vector_False_Negative_Rate(model, theta, X, Y)
    # True label=1 was in last two datapoints: want 1.0-prob[2:]
    arcomp_FNR = 1.0 - y_pred[2:]
    assert np.allclose(vector_FNR, arcomp_FNR)
    vector_TPR = zhat_funcs.vector_True_Positive_Rate(model, theta, X, Y)
    assert np.allclose(vector_TPR, 1.0 - arcomp_FNR)
    vector_TNR = zhat_funcs.vector_True_Negative_Rate(model, theta, X, Y)
    assert np.allclose(vector_TNR, 1.0 - arcomp_FPR)
    vector_ACC = zhat_funcs.vector_Accuracy(model, theta, X, Y, sub_regime=sub_regime)
    arcomp_ACC = np.array([0.5, 0.4378235, 0.62245933, 0.6791787])
    assert np.allclose(vector_ACC, arcomp_ACC)
    vector_ERR = zhat_funcs.vector_Error_Rate(model, theta, X, Y, sub_regime=sub_regime)
    arcomp_ERR = 1.0 - np.array([0.5, 0.4378235, 0.62245933, 0.6791787])
    assert np.allclose(vector_ERR, arcomp_ERR)


def test_multiclass_classification_measure_functions():
    # i = 4 datapoints
    # j = 2 features
    # k = 3 classes
    # labels are 0,1, or 2
    model = MultiClassLogisticRegressionModel()
    sub_regime = "multiclass_classification"
    Y = np.array([0, 0, 1, 1, 2, 2])  # length i, true labels
    X = np.array(
        [[0.0, 0.0], [-0.25, 0.0], [0.5, -0.5], [0.5, 0.5], [0.75, -0.25], [1.0, -1.0]]
    )  # i x j
    theta = np.array(
        [
            [0.0, 0.0, 0.0],  # intercept term - set to 0 for simplicity
            [1.0, 2.0, 3.0],
            [1.0, -1.0, 1.0],
        ]
    )  # (j+1,k), where j+1 to account for intercept
    y_pred = model.predict(theta, X)  # (i,k)

    # Accuracy
    ERR = objectives.Error_Rate(model, theta, X, Y, sub_regime=sub_regime)
    assert ERR == pytest.approx(1.0 - 0.36639504)

    # Vector accuracy and error rate
    vector_ACC = zhat_funcs.vector_Accuracy(model, theta, X, Y, sub_regime=sub_regime)
    arcomp_ACC = np.array(
        [0.33333333, 0.41922895, 0.54654939, 0.14024438, 0.49951773, 0.25949646]
    )
    assert np.allclose(vector_ACC, arcomp_ACC)

    # Vector accuracy
    vector_ERR = zhat_funcs.vector_Error_Rate(model, theta, X, Y, sub_regime=sub_regime)
    arcomp_ERR = 1.0 - np.array(
        [0.33333333, 0.41922895, 0.54654939, 0.14024438, 0.49951773, 0.25949646]
    )
    assert np.allclose(vector_ERR, arcomp_ERR)
    CM_array = np.zeros((3, 3))
    for class_index in [0, 1, 2]:
        # Will reuse these masks
        pos_mask = Y == class_index
        neg_mask = Y != class_index

        # Avg statistics
        PR = objectives.Positive_Rate(model, theta, X, Y, class_index=class_index)
        assert PR == np.mean(y_pred[:, class_index])
        NR = objectives.Negative_Rate(model, theta, X, Y, class_index=class_index)
        assert NR == pytest.approx(1.0 - PR)
        FPR = objectives.False_Positive_Rate(
            model, theta, X, Y, class_index=class_index
        )
        assert FPR == np.mean(y_pred[:, class_index][neg_mask])
        FNR = objectives.False_Negative_Rate(
            model, theta, X, Y, class_index=class_index
        )
        # true label was this class but predicted it was not this class
        assert FNR == np.mean(1.0 - y_pred[:, class_index][pos_mask])
        TPR = objectives.True_Positive_Rate(model, theta, X, Y, class_index=class_index)
        assert TPR == pytest.approx(1.0 - FNR)
        TNR = objectives.True_Negative_Rate(model, theta, X, Y, class_index=class_index)
        assert TNR == pytest.approx(1.0 - FPR)

        for l_k in [0, 1, 2]:
            res = objectives.confusion_matrix(
                model, theta, X, Y, l_i=class_index, l_k=l_k
            )
            CM_array[class_index][l_k] = res
            # The diagonals are the true positive for this class
            if l_k == class_index:
                assert res == TPR

        # Vector statistics
        vector_PR = zhat_funcs.vector_Positive_Rate(
            model, theta, X, Y, class_index=class_index
        )
        assert np.allclose(vector_PR, y_pred[:, class_index])
        vector_NR = zhat_funcs.vector_Negative_Rate(
            model, theta, X, Y, class_index=class_index
        )
        assert np.allclose(vector_NR, 1.0 - y_pred[:, class_index])
        vector_FPR = zhat_funcs.vector_False_Positive_Rate(
            model, theta, X, Y, class_index=class_index
        )
        arcomp_FPR = y_pred[:, class_index][neg_mask]
        assert np.allclose(vector_FPR, arcomp_FPR)
        vector_FNR = zhat_funcs.vector_False_Negative_Rate(
            model, theta, X, Y, class_index=class_index
        )
        arcomp_FNR = 1.0 - y_pred[:, class_index][pos_mask]
        assert np.allclose(vector_FNR, arcomp_FNR)
        vector_TPR = zhat_funcs.vector_True_Positive_Rate(
            model, theta, X, Y, class_index=class_index
        )
        assert np.allclose(vector_TPR, 1.0 - arcomp_FNR)
        vector_TNR = zhat_funcs.vector_True_Negative_Rate(
            model, theta, X, Y, class_index=class_index
        )
        assert np.allclose(vector_TNR, 1.0 - arcomp_FPR)
    # Checks on filled out confusion matrix
    CM_ans = np.array(
        [
            [0.37628114, 0.32991458, 0.29380427,],
            [0.17658777, 0.34339689, 0.48001534,],
            [0.07328825, 0.54720466, 0.3795071],
        ]
    )
    assert np.allclose(CM_array, CM_ans)
