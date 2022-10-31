import autograd.numpy as np   # Thinly-wrapped version of Numpy

from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedPytorchDataSet
from seldonian.models.pytorch_model import PytorchCNN
from seldonian.models import objectives
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)

import pandas as pd

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor

if __name__ == "__main__":
    torch.manual_seed(0)
    data_folder = '../../../notebooks/data'
    train_data = datasets.MNIST(
        root = data_folder,
        train = True,                         
        transform = ToTensor(), 
        download = False,            
    )
    N=500
    features = np.array(train_data.data[0:N].reshape(N,1,28,28),dtype='float32')/255.0
    labels = np.array(train_data.targets[0:N]) # these are 1D so don't need to reshape them
    label_column='label'

    dataset = SupervisedPytorchDataSet(
        features=features,
        labels=labels,
        label_column=label_column,
        meta_information=['img','label'])

    constraint_strs = ['ACC >= 0.75']
    deltas = [0.05] 

    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,deltas,regime='supervised_learning',
        sub_regime='multiclass_classification')

    model = PytorchCNN(input_dim=1,output_dim=1)
    initial_solution_fn = model.get_initial_weights
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime='multiclass_classification',
        primary_objective=objectives.multiclass_logistic_loss,
        use_builtin_primary_gradient_fn=False,
        initial_solution_fn=initial_solution_fn,
    )
    spec.optimization_hyperparams['lambda_init'] = np.array([0.5])
    spec.optimization_hyperparams['num_iters'] = 750
    spec.optimization_hyperparams['alpha_theta'] = 0.001

    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run(debug=True,write_cs_logfile=True)
    st_primary_objective = SA.evaluate_primary_objective(theta=solution,
        branch='safety_test')
    print("Primary objective evaluated on safety test:")
    print(st_primary_objective)