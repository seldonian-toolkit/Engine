# run_engine_custom_base_node.py
import autograd.numpy as np
from seldonian.utils.tutorial_utils import make_synthetic_regression_dataset
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import make_parse_trees_from_constraints
from seldonian.spec import SupervisedSpec
from seldonian.models.models import BoundedLinearRegressionModel
from seldonian.models import objectives


def main():
    """ Test that the gpa regression example runs 
    using the custom base node that calculates 
    CVaR alpha of the squared error. Make
    sure safety test passes and solution is correct.

    Check that the actual value of the constraint (not the bound)
    is also correctly calculated.
    """
    rseed=0
    np.random.seed(rseed) 
    constraint_strs = ['CVaRSQE <= 10.0']
    deltas = [0.1]

    numPoints = 75000
    dataset = make_synthetic_regression_dataset(
        numPoints,
        loc_X=0.0,
        loc_Y=0.0,
        sigma_X=1.0,
        sigma_Y=0.2,
        include_intercept_term=False,clipped=True)

    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,
        deltas)

    model = BoundedLinearRegressionModel()

    # Create spec object
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        sub_regime='regression',
        primary_objective=objectives.Mean_Squared_Error,
        use_builtin_primary_gradient_fn=False,
        custom_primary_gradient_fn=objectives.gradient_Bounded_Squared_Error,
        parse_trees=parse_trees,
        optimization_technique='gradient_descent',
        optimizer='adam',
        optimization_hyperparams={
            'lambda_init'   : np.array([0.5]),
            'alpha_theta'   : 0.01,
            'alpha_lamb'    : 0.01,
            'beta_velocity' : 0.9,
            'beta_rmsprop'  : 0.95,
            'num_iters'     : 50,
            'gradient_library': "autograd",
            'hyper_search'  : None,
            'verbose'       : True,
        }
    )

    # Run seldonian algorithm
    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run(write_cs_logfile=True,debug=False)
    if passed_safety:
        print("Passed safety test!")
        print(f"solution={solution}")
    else:
        print("Failed safety test")
        print("No solution found")

if __name__ == "__main__":
    main()