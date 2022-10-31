import autograd.numpy as np   # Thinly-wrapped version of Numpy

from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedPytorchDataSet
from seldonian.models.pytorch_model import PytorchCNN
from seldonian.models import objectives
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)
from seldonian.utils.io_utils import load_pickle
from seldonian.utils.plot_utils import plot_gradient_descent

import pandas as pd

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor

if __name__ == "__main__":
    f = "./logs/candidate_selection_log8.p"
    sol = load_pickle(f)
    plot_gradient_descent(sol,'log loss',save=True,
        savename='mnist_gradient_descent_N500_alpha0.001_ACCgt0.75.png')