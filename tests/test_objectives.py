import os
import pytest
import autograd.numpy as np

from seldonian.spec import (SupervisedSpec)
from seldonian.models import objectives
from seldonian.models.models import *
from seldonian.seldonian_algorithm import SeldonianAlgorithm

def test_binary_classification_measure_functions():
	# i = 4 datapoints
	# j = 2 features
	# labels are 0 or 1
	model = BinaryLogisticRegressionModel()
	sub_regime = 'binary_classification'
	Y = np.array([0,0,1,1]) # length i, true labels
	X = np.array([
		[0.0,0.0],
		[0.25,0.5],
		[0.5,1.0],
		[0.75,1.5]
		]) # i x j
	theta = np.array([0.0,-1.0,1.0]) # j+1 in length to account for intercept
	y_pred = model.predict(theta,X) # length i
	# Avg statistics
	PR = objectives.Positive_Rate(model,theta,X,Y)
	assert PR == pytest.approx(0.5909536328157614)
	NR = objectives.Negative_Rate(model,theta,X,Y)
	assert NR == pytest.approx(1.0-PR)
	FPR = objectives.False_Positive_Rate(model,theta,X,Y)
	# True label=0 was in first two datapoints. Avg(prob[1:3]) =~ 0.53
	assert FPR == pytest.approx(0.5310882504428991)
	FNR = objectives.False_Negative_Rate(model,theta,X,Y)
	# True label=1 was in last two datapoints. Avg(1.0-prob[2:]) =~ 0.35
	assert FNR == pytest.approx(0.3491809848113762)
	TPR = objectives.True_Positive_Rate(model,theta,X,Y)
	assert TPR == pytest.approx(1.0-FNR)
	TNR = objectives.True_Negative_Rate(model,theta,X,Y)
	assert TNR == pytest.approx(1.0-FPR)
	ACC = objectives.Accuracy(model,theta,X,Y,sub_regime=sub_regime)
	assert ACC == pytest.approx(0.5598653825)
	# Vector statistics 
	vector_PR = objectives.vector_Positive_Rate(model,theta,X,Y)
	assert np.allclose(vector_PR,y_pred)
	vector_NR = objectives.vector_Negative_Rate(model,theta,X,Y)
	assert np.allclose(vector_NR,1.0-y_pred)
	vector_FPR = objectives.vector_False_Positive_Rate(model,theta,X,Y)
	# True label=0 was in first two datapoints. prob[1:3] = [0.5,0.5621765]
	arcomp_FPR = y_pred[0:2]
	assert np.allclose(vector_FPR,arcomp_FPR)
	vector_FNR = objectives.vector_False_Negative_Rate(model,theta,X,Y)
	# True label=1 was in last two datapoints: want 1.0-prob[2:]
	arcomp_FNR = 1.0-y_pred[2:]
	assert np.allclose(vector_FNR,arcomp_FNR)
	vector_TPR = objectives.vector_True_Positive_Rate(model,theta,X,Y)
	assert np.allclose(vector_TPR,1.0-arcomp_FNR)
	vector_TNR = objectives.vector_True_Negative_Rate(model,theta,X,Y)
	assert np.allclose(vector_TNR,1.0-arcomp_FPR)
	vector_ACC = objectives.vector_Accuracy(model,theta,X,Y,sub_regime=sub_regime)
	arcomp_ACC = np.array([0.5, 0.4378235 , 0.62245933, 0.6791787 ])
	assert np.allclose(vector_ACC,arcomp_ACC)

def test_multiclass_classification_measure_functions():
	# i = 4 datapoints
	# j = 2 features
	# k = 3 classes
	# labels are 0,1, or 2
	model = MultiClassLogisticRegressionModel()
	sub_regime = 'multiclass_classification'
	Y = np.array([0,0,1,1,2,2]) # length i, true labels
	X = np.array([
		[0.0,0.0],
		[-0.25,0.0],
		[0.5,-0.5],
		[0.5,0.5],
		[0.75,-0.25],
		[1.0,-1.0]
		]) # i x j
	theta = np.array([
		[0.0,0.0,0.0], # intercept term - set to 0 for simplicity
		[1.0,2.0,3.0],
		[1.0,-1.0,1.0]]) # (j+1,k), where j+1 to account for intercept
	y_pred = model.predict(theta,X) # (i,k)

	# Accuracy
	ACC = objectives.Accuracy(model,theta,X,Y,sub_regime=sub_regime)
	assert ACC == pytest.approx(0.36639504)

	# Vector accuracy
	vector_ACC = objectives.vector_Accuracy(model,theta,X,Y,sub_regime=sub_regime)
	arcomp_ACC = np.array([0.33333333,0.41922895,0.54654939,0.14024438,0.49951773,0.25949646])
	assert np.allclose(vector_ACC,arcomp_ACC)

	for class_index in [0,1,2]:
		# Will reuse these masks
		pos_mask = Y == class_index
		neg_mask = Y != class_index

		# Avg statistics
		PR = objectives.Positive_Rate(model,theta,X,Y,
			class_index=class_index)
		assert PR == np.mean(y_pred[:,class_index])
		NR = objectives.Negative_Rate(model,theta,X,Y,
			class_index=class_index)
		assert NR == pytest.approx(1.0-PR)
		FPR = objectives.False_Positive_Rate(model,theta,X,Y,
			class_index=class_index)
		assert FPR == np.mean(y_pred[:,class_index][neg_mask])
		FNR = objectives.False_Negative_Rate(model,theta,X,Y,
			class_index=class_index)
		# true label was this class but predicted it was not this class
		assert FNR == np.mean(1.0-y_pred[:,class_index][pos_mask])
		TPR = objectives.True_Positive_Rate(model,theta,X,Y,
			class_index=class_index)
		assert TPR == pytest.approx(1.0 - FNR)
		TNR = objectives.True_Negative_Rate(model,theta,X,Y,
			class_index=class_index)
		assert TNR == pytest.approx(1.0 - FPR)
	
		# Vector statistics 
		vector_PR = objectives.vector_Positive_Rate(
			model,theta,X,Y,
			class_index=class_index)
		assert np.allclose(vector_PR,y_pred[:,class_index])
		vector_NR = objectives.vector_Negative_Rate(
			model,theta,X,Y,
			class_index=class_index)
		assert np.allclose(vector_NR,1.0-y_pred[:,class_index])
		vector_FPR = objectives.vector_False_Positive_Rate(
			model,theta,X,Y,
			class_index=class_index)
		arcomp_FPR = y_pred[:,class_index][neg_mask]
		assert np.allclose(vector_FPR,arcomp_FPR)
		vector_FNR = objectives.vector_False_Negative_Rate(
			model,theta,X,Y,
			class_index=class_index)
		arcomp_FNR = 1.0-y_pred[:,class_index][pos_mask]
		assert np.allclose(vector_FNR,arcomp_FNR)
		vector_TPR = objectives.vector_True_Positive_Rate(
			model,theta,X,Y,
			class_index=class_index)
		assert np.allclose(vector_TPR,1.0-arcomp_FNR)
		vector_TNR = objectives.vector_True_Negative_Rate(
			model,theta,X,Y,
			class_index=class_index)
		assert np.allclose(vector_TNR,1.0-arcomp_FPR)