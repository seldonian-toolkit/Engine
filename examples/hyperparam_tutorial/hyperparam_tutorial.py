import pickle
import tqdm
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import random
from seldonian.models.models import LinearRegressionModel
from seldonian.spec import SupervisedSpec,HyperparameterSelectionSpec
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.hyperparam_search import HyperparamSearch
from seldonian.utils.tutorial_utils import (
    make_synthetic_regression_dataset)
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)

if __name__ == "__main__":
    results_dir = "test"
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

    # 4. Create specs object.
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime='regression',
    )
    # 4. Do hyperparameter search.
    all_frac_data_in_safety = [0.1, 0.3, 0.5, 0.7, 0.9]
    hyperparam_spec = HyperparameterSelectionSpec(
        n_bootstrap_trials=100,
        all_frac_data_in_safety=all_frac_data_in_safety,
        n_bootstrap_workers=6,
        use_bs_pools=True,
        confidence_interval_type=None
    )
    
    HS = HyperparamSearch(spec=spec, hyperparam_spec=hyperparam_spec, results_dir=results_dir)
    
    # Test create_dataset.
    candidate_dataset, safety_dataset = HS.create_dataset(
            HS.dataset, all_frac_data_in_safety[0], results_dir)
    
    frac_data_in_safety, candidate_dataset, safety_dataset, ran_new_bs_trials = \
            HS.find_best_frac_data_in_safety()

    # 5. Update spec to have new frac_data_in_safety.
    spec.frac_data_in_safety = frac_data_in_safety
    print("Best frac_data_in_safety:", frac_data_in_safety) # 0.7

    # 6. Run core to get Seldonian algorithm solution.
    SA = SeldonianAlgorithm(spec)
    passed_safety, solution = SA.run()
    print(passed_safety, solution) # True, [0.17256748 0.17382125]
