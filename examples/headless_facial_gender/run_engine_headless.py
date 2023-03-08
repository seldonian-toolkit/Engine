# tensorflow_mnist.py
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
import os

from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedDataSet
from seldonian.utils.io_utils import load_pickle,save_pickle
from model_head import PytorchFacialRecogHead
from seldonian.models import objectives
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
	make_parse_trees_from_constraints)

import torch

if __name__ == "__main__":
	torch.manual_seed(0)
	regime='supervised_learning'
	sub_regime='classification'
	N=23700
	savename_features = './facial_gender_latent_features.pkl'
	savename_labels = './facial_gender_labels.pkl'
	savename_sensitive_attrs = './sensitive_attrs.pkl'
	features = load_pickle(savename_features)
	labels = load_pickle(savename_labels)
	sensitive_attrs = load_pickle(savename_sensitive_attrs)
	assert len(features) == N
	assert len(labels) == N
	assert len(sensitive_attrs) == N
	frac_data_in_safety = 0.5
	sensitive_col_names = ['M','F']

	meta_information = {}
	meta_information['feature_col_names'] = ['img']
	meta_information['label_col_names'] = ['label']
	meta_information['sensitive_col_names'] = sensitive_col_names
	meta_information['sub_regime'] = sub_regime
	
	print("Making SupervisedDataSet...")
	dataset = SupervisedDataSet(
		features=features,
		labels=labels,
		sensitive_attrs=sensitive_attrs,
		num_datapoints=N,
		meta_information=meta_information)

	constraint_strs = ['min((ACC | [M])/(ACC | [F]),(ACC | [F])/(ACC | [M])) >= 0.8']
	deltas = [0.05] 
	print("Making parse trees for constraint(s):")
	print(constraint_strs," with deltas: ", deltas)
	parse_trees = make_parse_trees_from_constraints(
		constraint_strs,deltas,regime=regime,
		sub_regime=sub_regime,columns=sensitive_col_names)

	device = torch.device("cpu")
	model = PytorchFacialRecogHead(device)

	initial_solution_fn = model.get_model_params
	spec = SupervisedSpec(
		dataset=dataset,
		model=model,
		parse_trees=parse_trees,
		frac_data_in_safety=frac_data_in_safety,
		primary_objective=objectives.binary_logistic_loss,
		use_builtin_primary_gradient_fn=False,
		sub_regime=sub_regime,
		initial_solution_fn=initial_solution_fn,
		optimization_technique='gradient_descent',
		optimizer='adam',
		optimization_hyperparams={
			'lambda_init'   : np.array([0.5]),
			'alpha_theta'   : 0.01,
			'alpha_lamb'    : 0.01,
			'beta_velocity' : 0.9,
			'beta_rmsprop'  : 0.95,
			'use_batches'   : False,
			'num_iters'     : 1200,
			'gradient_library': "autograd",
			'hyper_search'  : None,
			'verbose'       : False,
		},
		batch_size_safety=2000
	)
	save_pickle('./spec.pkl',spec,verbose=True)
	SA = SeldonianAlgorithm(spec)

	passed_safety,solution = SA.run(debug=True,write_cs_logfile=True)
	if passed_safety:
		print("Passed safety test")
		st_primary_objective = SA.evaluate_primary_objective(theta=solution,
		branch='safety_test')
		print("Primary objective evaluated on safety test:")
		print(st_primary_objective)
		upper_bounds_dict = SA.get_st_upper_bounds()
		print("upper_bounds_dict:")
		print(upper_bounds_dict)
	else:
		print("Failed safety test")