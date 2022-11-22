# tensorflow_mnist.py
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedDataSet
from seldonian.utils.io_utils import load_pickle
from seldonian.models.tensorflow_cnn import TensorFlowCNN
from seldonian.models import objectives
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
	make_parse_trees_from_constraints)

import tensorflow as tf

if __name__ == "__main__":
	tf.random.set_seed(42)
	regime='supervised_learning'
	sub_regime='multiclass_classification'
	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()    
	x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
	x_train=(x_train / 255.0).astype('float32')
	x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
	x_test=(x_test/255.0).astype('float32')
	# Combine x_train and x_test into a single feature array and take first N of them
	N=70000
	features = np.vstack([x_train,x_test])[0:N]
	# Combine y_train and y_test into a single label array
	labels = np.hstack([y_train,y_test])[0:N]
	 
	assert len(features) == N
	assert len(labels) == N
	frac_data_in_safety = 0.5

	meta_information = {}
	meta_information['feature_col_names'] = ['img']
	meta_information['label_col_names'] = ['label']
	meta_information['sensitive_col_names'] = []
	meta_information['sub_regime'] = sub_regime

	dataset = SupervisedDataSet(
		features=features,
		labels=labels,
		sensitive_attrs=[],
		num_datapoints=N,
		meta_information=meta_information)

	constraint_strs = ['ACC >= 0.9']
	deltas = [0.05] 

	parse_trees = make_parse_trees_from_constraints(
		constraint_strs,deltas,regime=regime,
		sub_regime=sub_regime)

	model = TensorFlowCNN()

	initial_solution_fn = model.get_initial_weights
	theta_file = '../../../notebooks/best_theta_mnist_tensorflow.pkl'
	best_theta = load_pickle(theta_file)
	# print("loaded best theta:")
	# print(best_theta)
	# initial_solution_fn = lambda *args: initial_theta
	batch_size_safety=5000
	spec = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=objectives.multiclass_logistic_loss,
		use_builtin_primary_gradient_fn=False,
		sub_regime=sub_regime,
		initial_solution_fn=initial_solution_fn,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.001,
			'alpha_lamb'    : 0.01,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'use_batches'   : True,
			'batch_size'    : 150,
			'n_epochs'      : 5,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : True,
		},
		batch_size_safety=batch_size_safety
	)
	# save_pickle('./spec.pkl',spec)
	SA = SeldonianAlgorithm(spec)
	# initial_weights = initial_solution_fn(dataset,model)
	# SA.initial_solution = best_theta
	# print("random initial weights are:")
	# print(initial_weights)
	# model.update_model_params(initial_weights)

	# passed_safety,solution = SA.run(debug=True,write_cs_logfile=True)
	# f_cs_best = SA.evaluate_primary_objective(branch='candidate_selection',theta=best_theta)
	passed_safety,solution = SA.run_safety_test(
		candidate_solution=best_theta,batch_size_safety=batch_size_safety,debug=True)
	# f_st_best = SA.evaluate_primary_objective(branch='safety_test',theta=best_theta)
	# print("f_st_best:")
	# print(f_st_best)
	# print("f_cs_best,f_st_best")
	# print(f_cs_best,f_st_best)
	# if passed_safety:
	# 	print("Passed safety test.")
	# else:
	# 	print("Failed safety test")
	# st_primary_objective = SA.evaluate_primary_objective(theta=solution,
	# 	branch='safety_test')
	# print("Primary objective evaluated on safety test:")
	# print(st_primary_objective)