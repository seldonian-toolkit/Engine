import autograd.numpy as np  # Thinly-wrapped version of Numpy
from seldonian.models.sklearn_lr import SkLearnLinearRegressor
from seldonian.spec import SupervisedSpec
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.tutorial_utils import make_synthetic_regression_dataset
from seldonian.parse_tree.parse_tree import make_parse_trees_from_constraints

### Begin tests


def test_sklearn_linear_regressor():
    # dummy data for linear regression

    np.random.seed(0)
    num_points = 1000
    # 1. Define the data - X ~ N(0,1), Y ~ X + N(0,1)
    dataset = make_synthetic_regression_dataset(num_points=num_points)

    # 2. Create parse trees from the behavioral constraints
    # constraint strings:
    constraint_strs = ["Mean_Squared_Error >= 1.25", "Mean_Squared_Error <= 2.0"]
    # confidence levels:
    deltas = [0.1, 0.1]

    parse_trees = make_parse_trees_from_constraints(constraint_strs, deltas)

    # 3. Define the underlying machine learning model
    model = SkLearnLinearRegressor()
    assert model.has_intercept == True

    """4. Create a spec object, using some
	hidden defaults we won't worry about here
	"""
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
    )
    spec.use_builtin_primary_gradient_fn = True
    # 5. Run seldonian algorithm using the spec object
    SA = SeldonianAlgorithm(spec)

    passed_safety, solution = SA.run(debug=True, write_cs_logfile=True)
    assert passed_safety == True
    target_solution = np.array([0.16911355, 0.1738146])
    assert np.allclose(solution, target_solution)
