import os, sys
import autograd.numpy as np  # Thinly-wrapped version of Numpy

from seldonian.seldonian_algorithm import seldonian_algorithm
from seldonian.utils.io_utils import load_pickle

if __name__ == '__main__':
    specfile = '/home/james/seldonian_library_repos/Engine/spec.pkl'
    spec = load_pickle(specfile)
    spec.use_builtin_primary_gradient_fn = False
    spec.optimization_hyperparams['num_iters'] = 20
    spec.optimization_hyperparams['alpha_theta'] = 0.05
    spec.optimization_hyperparams['alpha_lamb'] = 0.05
    # spec.regularization_hyperparams['reg_coef'] = 0.1
    passed_safety, candidate_solution = seldonian_algorithm(spec)
    print(passed_safety, candidate_solution)
