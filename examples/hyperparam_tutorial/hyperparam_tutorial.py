import pickle
import tqdm
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from seldonian.models.models import LinearRegressionModel
from seldonian.spec import SupervisedSpec
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.hyperparam_search import HyperparamSearch
from seldonian.utils.tutorial_utils import (
    make_synthetic_regression_dataset)
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)

if __name__ == "__main__":
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
    all_frac_data_in_candidate_selection = [0.1, 0.3, 0.5, 0.7, 0.9]
    HS = HyperparamSearch(spec, all_frac_data_in_candidate_selection)
    frac_data_in_safety, candidate_dataset, safety_dataset = HS.find_best_hyperparams()
    n_safety = len(safety_dataset.df)

    # 5. Run core to get Seldonian algorithm solution.
    passed_safety, solution = HS.run_core(candidate_dataset, safety_dataset, n_safety, frac_data_in_safety)
    print("Best frac_data_in_safety:", frac_data_in_safety)
    print(passed_safety, solution)
