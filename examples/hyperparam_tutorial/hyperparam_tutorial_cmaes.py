import os
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import tqdm
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import random
from seldonian.models.models import LinearRegressionModel
from seldonian.spec import SupervisedSpec,HyperparameterSelectionSpec
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.hyperparam_search import HyperparamSearch,HyperSchema
from seldonian.utils.tutorial_utils import (
    make_synthetic_regression_dataset)
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)

if __name__ == "__main__":
    results_dir = "results_cmaes"
    os.makedirs(results_dir,exist_ok=True)
    np.random.seed(0)
    num_points = 1000

    # 1. Define the data - X ~ N(0,1), Y ~ X + N(0,1)
    dataset = make_synthetic_regression_dataset(num_points=num_points)

    # 2. Create parse trees from the behavioral constraints constraint strings:
    constraint_strs = ['Mean_Squared_Error >= 1.25','Mean_Squared_Error <= 2.0']
    # confidence levels: 
    deltas = [0.1,0.1] 

    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,deltas)

    # 3. Define underlying machine learning model.
    model = LinearRegressionModel()

    # 4. Create spec object.
    frac_data_in_safety=0.6
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime='regression',
        frac_data_in_safety=frac_data_in_safety,
    )
    spec.verbose=True
    # 4. Do hyperparameter search using CMA-ES

    hyper_schema = HyperSchema(
        tuning_method="CMA-ES",
        hyper_dict={
            "alpha_theta": {
                "initial_value": 0.005,
                "hyper_type": "optimization",
                "dtype": "float",
                "min_val": 0.0001,
                "max_val": 0.1
                },
            "num_iters": {
                "initial_value": 100,
                "hyper_type": "optimization",
                "dtype": "int",
                "min_val": 10,
                "max_val": 10000,
                }
            }
    )
    n_bootstrap_trials = 10
    n_bootstrap_workers = 1
    use_bs_pools=True
    HS_spec = HyperparameterSelectionSpec(
            hyper_schema=hyper_schema,
            n_bootstrap_trials=n_bootstrap_trials,
            n_bootstrap_workers=n_bootstrap_workers,
            use_bs_pools=use_bs_pools,
            confidence_interval_type=None
    )
    
    
    HS = HyperparamSearch(spec=spec, hyperparam_spec=HS_spec, results_dir=results_dir)

    # For a given frac data in safety find best combo of other hyperparams
    best_hyperparam_setting, best_hyperparam_spec = HS.find_best_hyperparameters(
            frac_data_in_safety=frac_data_in_safety,
            tuning_method="CMA-ES")
    print(best_hyperparam_setting)
    # expected_best_hyperparam_settuion) # True, [0.17256748 0.17382125]
