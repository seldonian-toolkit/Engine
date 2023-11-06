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
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": 0.5,
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 200,
            "gradient_library": "autograd",
            "use_batches": False,
            "hyper_search": None,
            "verbose": False,
        },
    )
    spec.verbose=False
    # 4. Do hyperparameter search using CMA-ES

    hyper_schema = HyperSchema(
        hyper_dict={
            "alpha_theta": {
                "initial_value":0.005,
                "min_val": 0.0001,
                "max_val": 0.1,
                "dtype": "float",
                "hyper_type":"optimization",
                "search_distribution": "log-uniform",
                "tuning_method": "CMA-ES"
                },
            "alpha_lamb": {
                "initial_value":0.005,
                "min_val": 0.0001,
                "max_val": 0.1,
                "dtype": "float",
                "hyper_type":"optimization",
                "search_distribution": "log-uniform",
                "tuning_method": "CMA-ES"
                },
            "bound_inflation_factor": {
                "initial_value":[1,1],
                "min_val": 0.1,
                "max_val": 4,
                "dtype": "float",
                "hyper_type":"SA",
                "search_distribution": "uniform",
                "tuning_method": "CMA-ES"
                },
            "num_iters": {
                "values": [100,200],
                "hyper_type": "optimization",
                "tuning_method": "grid_search"
                }
            }
    )
    n_bootstrap_trials = 5
    n_bootstrap_workers = 10
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
    print()
    print("Finished!")
    print("best hyperparameters found:")
    print(best_hyperparam_setting)
