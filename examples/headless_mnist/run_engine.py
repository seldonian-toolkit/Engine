# pytorch_mnist.py
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedDataSet
from cnnhead_model import CNNHead
from seldonian.models import objectives
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)
from seldonian.utils.io_utils import load_pickle,save_pickle

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor


if __name__ == "__main__":
    torch.manual_seed(0)
    regime='supervised_learning'
    sub_regime='multiclass_classification'
    
    # load 70,000 data points
    data_file = './mnist_latent_features.pkl'
    label_file = './mnist_labels.pkl'
    print("loading data...")
    features = load_pickle(data_file)
    labels = load_pickle(label_file) 
    print("done")
    print()
    N=70000
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

    constraint_strs = ['ACC >= 0.95']
    deltas = [0.05] 

    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,deltas,regime=regime,
        sub_regime=sub_regime)
    # device = torch.device("mps")
    device = torch.device("cpu")
    model = CNNHead(device)

    initial_solution_fn = model.get_model_params
    
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
    )
    save_pickle('headless_mnist_spec.pkl',spec,verbose=True)

    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run(debug=True,write_cs_logfile=True)
    if passed_safety:
        print("Passed safety test.")
    else:
        print("Failed safety test")
    # st_primary_objective = SA.evaluate_primary_objective(theta=solution,
    #     branch='safety_test')
    # print("Primary objective evaluated on safety test:")
    # print(st_primary_objective)