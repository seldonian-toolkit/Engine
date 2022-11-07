import autograd.numpy as np   # Thinly-wrapped version of Numpy

from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedDataSet
from seldonian.models.pytorch_cnn import PytorchCNN
from seldonian.models import objectives
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)
from seldonian.utils.io_utils import load_pickle

import pandas as pd

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

# def manual_initial_weights(*args):
#     f = "/Users/ahoag/beri/code/engine-repo-dev/examples/pytorch_mnist/logs/candidate_selection_log10.p"
#     sol = load_pickle(f)
#     return sol['theta_vals'][-1]

if __name__ == "__main__":
    torch.manual_seed(0)
    regime='supervised_learning'
    sub_regime='multiclass_classification'
    data_folder = '../../../notebooks/data'
    train_data = datasets.MNIST(
        root = data_folder,
        train = True,                         
        transform = ToTensor(), 
        download = False,            
    )
    N=60000
    frac_data_in_safety = 0.6
    features = np.array(train_data.data[0:N].reshape(N,1,28,28),dtype='float32')/255.0
    labels = np.array(train_data.targets[0:N]) # these are 1D so don't need to reshape them
    label_column='label'

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
    device = torch.device("mps")
    model = PytorchCNN(device)

    initial_solution_fn = model.get_initial_weights
    
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
            'batch_size'    : 100,
            'n_epochs'      : 5,
            'gradient_library': "autograd",
            'hyper_search'  : None,
            'verbose'       : True,
        },
    )

    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run(debug=True,write_cs_logfile=True)
    st_primary_objective = SA.evaluate_primary_objective(theta=solution,
        branch='safety_test')
    print("Primary objective evaluated on safety test:")
    print(st_primary_objective)