import os
import pytest
import importlib
import autograd.numpy as np
import pandas as pd

from seldonian.utils.io_utils import load_json, load_pickle
from seldonian.utils.tutorial_utils import (
    make_synthetic_regression_dataset,
    generate_data,
)
from seldonian.parse_tree.parse_tree import ParseTree, make_parse_trees_from_constraints
from seldonian.dataset import DataSetLoader, SupervisedDataSet, RLDataSet

from seldonian.spec import Spec, RLSpec, SupervisedSpec, createSupervisedSpec
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.models.models import *
from seldonian.models import objectives
from seldonian.RL.RL_model import RL_model

import matplotlib.pyplot as plt

### Begin tests


def test_base_node_bound_methods_updated(gpa_regression_dataset):
    rseed = 99
    np.random.seed(rseed)
    constraint_strs = ["Mean_Squared_Error - 5.0", "2.0 - Mean_Squared_Error"]
    deltas = [0.05, 0.05]
    (dataset, model, primary_objective, parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )
    assert (
        parse_trees[0].base_node_dict["Mean_Squared_Error"]["bound_method"] == "ttest"
    )
    assert (
        parse_trees[1].base_node_dict["Mean_Squared_Error"]["bound_method"] == "ttest"
    )
    base_node_bound_method_dict = {
        "Mean_Squared_Error - 5.0": {"Mean_Squared_Error": "manual"},
        "2.0 - Mean_Squared_Error": {"Mean_Squared_Error": "random"},
    }
    frac_data_in_safety = 0.6

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=False,
        base_node_bound_method_dict=base_node_bound_method_dict,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="barrier_function",
        optimizer="Powell",
        optimization_hyperparams={
            "maxiter": 1000,
            "seed": rseed,
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Build SA object and verify that the bound method was updated
    SA = SeldonianAlgorithm(spec)
    assert (
        parse_trees[0].base_node_dict["Mean_Squared_Error"]["bound_method"] == "manual"
    )
    assert (
        parse_trees[1].base_node_dict["Mean_Squared_Error"]["bound_method"] == "random"
    )


def test_not_enough_data(simulated_regression_dataset):
    # dummy data for linear regression

    constraint_strs = ["Mean_Squared_Error - 2.0"]
    deltas = [0.5]
    numPoints = 3
    (dataset, model, primary_objective, parse_trees) = simulated_regression_dataset(
        constraint_strs, deltas, numPoints=numPoints
    )
    frac_data_in_safety = 0.6

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    # Create spec object
    # Will warn because of initial solution trying to fit with not enough data
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=objectives.Mean_Squared_Error,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.01,
            "alpha_lamb": 0.01,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 200,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )
    warning_msg = "Warning: not enough data to run the Seldonian algorithm."
    with pytest.warns(UserWarning, match=warning_msg) as excinfo:
        SA = SeldonianAlgorithm(spec)
        passed_safety, solution = SA.run()

    spec_zeros = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=objectives.Mean_Squared_Error,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=lambda x, y: np.zeros(2),
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.01,
            "alpha_lamb": 0.01,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 200,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )
    warning_msg = "Warning: not enough data to run the Seldonian algorithm."
    with pytest.warns(UserWarning, match=warning_msg) as excinfo:
        SA = SeldonianAlgorithm(spec_zeros)
        passed_safety, solution = SA.run()


def test_data_as_lists(simulated_regression_dataset_aslists):
    # dummy data for linear regression

    constraint_strs = ["Mean_Squared_Error - 2.0"]
    deltas = [0.5]
    numPoints = 1000
    (
        dataset,
        model,
        primary_objective,
        parse_trees,
    ) = simulated_regression_dataset_aslists(
        constraint_strs, deltas, numPoints=numPoints
    )
    frac_data_in_safety = 0.6

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    # Create spec object
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=objectives.Mean_Squared_Error,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.01,
            "alpha_lamb": 0.01,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 200,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    SA = SeldonianAlgorithm(spec)
    candidate_features = SA.candidate_dataset.features
    candidate_labels = SA.candidate_dataset.labels
    assert type(candidate_features) == list
    assert type(candidate_labels) == np.ndarray

    with pytest.raises(NotImplementedError) as excinfo:
        SA = SeldonianAlgorithm(spec)
        passed_safety, solution = SA.run()
    error_str = (
        "This function is not supported when features are in a list. "
        "Convert features to a numpy array if possible or use autodiff "
        " to get the gradient."
    )
    assert str(excinfo.value) == error_str

    (
        dataset,
        model,
        primary_objective,
        parse_trees,
    ) = simulated_regression_dataset_aslists(
        constraint_strs, deltas, numPoints=numPoints
    )
    # Create spec object using autodiff
    spec2 = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=objectives.Mean_Squared_Error,
        use_builtin_primary_gradient_fn=False,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.01,
            "alpha_lamb": 0.01,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 10,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    SA2 = SeldonianAlgorithm(spec2)
    candidate_features = SA2.candidate_dataset.features
    candidate_labels = SA2.candidate_dataset.labels
    assert type(candidate_features) == list
    assert type(candidate_labels) == np.ndarray

    passed_safety, solution = SA2.run()
    assert passed_safety == True
    array_to_compare = np.array([0.02483889, 0.98311923, 0.02349485])
    assert np.allclose(solution, array_to_compare)


def test_bad_optimizer(gpa_regression_dataset):
    """Test that attempting to use an optimizer
    or optimization_technique that is not supported
    raises an error"""

    rseed = 99
    np.random.seed(rseed)
    constraint_strs = ["Mean_Squared_Error - 2.0"]
    deltas = [0.05]
    (dataset, model, primary_objective, parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )

    frac_data_in_safety = 0.6

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    bad_optimizer = "bad-optimizer"
    for optimization_technique in ["barrier_function", "gradient_descent"]:
        bad_spec = SupervisedSpec(
            dataset=dataset,
            model=model,
            parse_trees=parse_trees,
            sub_regime="regression",
            frac_data_in_safety=frac_data_in_safety,
            primary_objective=primary_objective,
            use_builtin_primary_gradient_fn=False,
            initial_solution_fn=initial_solution_fn,
            optimization_technique=optimization_technique,
            optimizer=bad_optimizer,
            optimization_hyperparams={
                "maxiter": 1000,
                "seed": rseed,
                "hyper_search": None,
                "verbose": True,
            },
        )

        # Run seldonian algorithm
        with pytest.raises(NotImplementedError) as excinfo:
            SA = SeldonianAlgorithm(bad_spec)
            passed_safety, solution = SA.run()
        error_str = "Optimizer: bad-optimizer is not supported"
        assert error_str in str(excinfo.value)

    bad_optimization_technique = "bad-opt-technique"

    bad_spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=False,
        initial_solution_fn=initial_solution_fn,
        optimization_technique=bad_optimization_technique,
        optimizer="adam",
        optimization_hyperparams={
            "maxiter": 1000,
            "seed": rseed,
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Run seldonian algorithm
    with pytest.raises(NotImplementedError) as excinfo:
        SA = SeldonianAlgorithm(bad_spec)
        passed_safety, solution = SA.run()
    error_str = "Optimization technique: bad-opt-technique is not implemented"
    assert error_str in str(excinfo.value)


def test_phil_custom_base_node(gpa_regression_dataset):
    """Test that the gpa regression example runs
    using Phil's custom base node. Make
    sure safety test passes and solution is correct.
    """
    rseed = 0
    np.random.seed(rseed)
    # constraint_strs = ['Mean_Squared_Error - 2.0']
    constraint_strs = ["MED_MF - 0.1"]
    deltas = [0.05]

    (dataset, model, primary_objective, parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )

    frac_data_in_safety = 0.6

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    # Create spec object
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.01,
            "alpha_lamb": 0.01,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 10,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Run seldonian algorithm
    SA = SeldonianAlgorithm(spec)
    passed_safety, solution = SA.run(debug=True)
    assert passed_safety == True
    array_to_compare = np.array(
        [
            0.42523186,
            -0.00285919,
            -0.00202504,
            -0.00241554,
            -0.00234768,
            -0.00258539,
            0.01924093,
            0.01865392,
            -0.00308652,
            -0.00244911,
        ]
    )

    assert np.allclose(solution, array_to_compare)


def test_cvar_custom_base_node():
    """Test that the gpa regression example runs
    using the custom base node that calculates
    CVaR alpha of the squared error. Make
    sure safety test passes and solution is correct.

    Check that the actual value of the constraint (not the bound)
    is also correctly calculated.
    """
    from seldonian.models.models import BoundedLinearRegressionModel

    rseed = 0
    np.random.seed(rseed)
    constraint_strs = ["CVaRSQE <= 50.0"]
    deltas = [0.1]

    numPoints = 2500
    dataset = make_synthetic_regression_dataset(
        numPoints, loc_X=0.0, loc_Y=0.0, sigma_X=1.0, sigma_Y=0.2, clipped=True
    )

    parse_trees = make_parse_trees_from_constraints(constraint_strs, deltas)

    model = BoundedLinearRegressionModel()

    # Create spec object
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        sub_regime="regression",
        primary_objective=objectives.Mean_Squared_Error,
        use_builtin_primary_gradient_fn=False,
        custom_primary_gradient_fn=objectives.gradient_Bounded_Squared_Error,
        parse_trees=parse_trees,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.01,
            "alpha_lamb": 0.01,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 5,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Run seldonian algorithm
    SA = SeldonianAlgorithm(spec)
    passed_safety, solution = SA.run(debug=True)
    assert passed_safety == True
    solution_to_compare = np.array([-0.07257342, 0.07182381])
    assert np.allclose(solution, solution_to_compare)

    # Make sure we can evaluate constraint as well
    pt = parse_trees[0]
    pt.evaluate_constraint(
        theta=solution,
        tree_dataset_dict={"all": dataset},
        model=model,
        regime="supervised_learning",
        branch="safety_test",
        sub_regime="regression",
    )
    assert pt.root.value == pytest.approx(-47.163772762)


def test_cvar_lower_bound():
    """The normal constraint only uses
    the CVAR upper bound because we want CVAR < some value.
    Test that the lower bound also works

    Check that the actual value of the constraint (not the bound)
    is also correctly calculated.
    """
    from seldonian.models.models import BoundedLinearRegressionModel

    rseed = 0
    np.random.seed(rseed)
    constraint_strs = ["CVaRSQE >= 5.0"]
    deltas = [0.1]

    numPoints = 1000
    dataset = make_synthetic_regression_dataset(
        numPoints, loc_X=0.0, loc_Y=0.0, sigma_X=1.0, sigma_Y=0.2, clipped=True
    )

    parse_trees = make_parse_trees_from_constraints(constraint_strs, deltas)

    model = BoundedLinearRegressionModel()

    # Create spec object
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        sub_regime="regression",
        primary_objective=objectives.Mean_Squared_Error,
        use_builtin_primary_gradient_fn=False,
        custom_primary_gradient_fn=objectives.gradient_Bounded_Squared_Error,
        parse_trees=parse_trees,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.01,
            "alpha_lamb": 0.01,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 10,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Run seldonian algorithm
    SA = SeldonianAlgorithm(spec)
    passed_safety, solution = SA.run()
    assert passed_safety == True
    solution_to_compare = np.array([-0.15426298, -0.15460036])
    assert np.allclose(solution, solution_to_compare)


def test_gpa_data_regression_multiple_constraints(gpa_regression_dataset):
    """Test that the gpa regression example runs
    with a two constraints using gradient descent. Make
    sure safety test passes and solution is correct.
    """
    # Load metadata
    rseed = 0
    np.random.seed(rseed)
    constraint_strs = ["Mean_Squared_Error - 5.0", "2.0 - Mean_Squared_Error"]
    deltas = [0.05, 0.1]

    (dataset, model, primary_objective, parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )

    frac_data_in_safety = 0.6

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    # Create spec object
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5, 0.5]),
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 200,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Run seldonian algorithm
    SA = SeldonianAlgorithm(spec)
    passed_safety, solution = SA.run()
    assert passed_safety == True
    array_to_compare = np.array(
        [
            4.18121191e-01,
            7.65218366e-05,
            8.68827231e-04,
            4.96795941e-04,
            5.40624536e-04,
            3.35472715e-04,
            2.10383120e-03,
            1.52231771e-03,
            -1.46634476e-04,
            4.67094023e-04,
        ]
    )
    assert np.allclose(solution, array_to_compare)


def test_gpa_data_regression_custom_constraint(gpa_regression_dataset):
    """Test that the gpa regression example runs
    using Phil's custom base node: MED_MF. Make
    sure safety test passes and solution is correct.
    """
    rseed = 0
    np.random.seed(rseed)
    constraint_strs = ["MED_MF - 0.2"]
    deltas = [0.05]

    (dataset, model, primary_objective, parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )

    frac_data_in_safety = 0.6

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    # Create spec object
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 200,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Run seldonian algorithm
    SA = SeldonianAlgorithm(spec)
    passed_safety, solution = SA.run()
    assert passed_safety == True
    array_to_compare = np.array(
        [
            0.42155706,
            -0.00153405,
            -0.00069985,
            -0.00109037,
            -0.00102248,
            -0.00126023,
            0.01056612,
            0.00997911,
            -0.0017614,
            -0.00112394,
        ]
    )

    assert np.allclose(solution, array_to_compare)


def test_gpa_data_regression_addl_datasets(gpa_regression_addl_datasets):
    """Test that the gpa regression example runs
    when using a different dataset for the base nodes compared to the primary objective
    """
    # Load metadata
    rseed = 0
    np.random.seed(rseed)
    constraint_strs = ["Mean_Squared_Error - 5.0", "2.0 - Mean_Squared_Error"]
    deltas = [0.05, 0.1]

    (
        primary_dataset,
        additional_datasets,
        model,
        primary_objective,
        parse_trees,
    ) = gpa_regression_addl_datasets(constraint_strs=constraint_strs, deltas=deltas)

    frac_data_in_safety = 0.6

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    # Create spec object
    spec = SupervisedSpec(
        dataset=primary_dataset,
        additional_datasets=additional_datasets,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5, 0.5]),
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 200,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Set up SA object
    SA = SeldonianAlgorithm(spec)
    # Ensure that the candidate and safety datasets were created within the additional_datasets object
    for pt in spec.parse_trees:
        base_nodes_this_tree = list(pt.base_node_dict.keys())
        for bn in base_nodes_this_tree:
            assert "candidate_dataset" in additional_datasets[pt.constraint_str][bn]
            assert "safety_dataset" in additional_datasets[pt.constraint_str][bn]
            bn_dataset = additional_datasets[pt.constraint_str][bn]["dataset"]
            assert bn_dataset.num_datapoints == int(
                round(spec.dataset.num_datapoints * 0.8)
            )
            cd = additional_datasets[pt.constraint_str][bn]["candidate_dataset"]
            assert cd.num_datapoints == int(
                round((1.0 - frac_data_in_safety) * 0.8 * spec.dataset.num_datapoints)
            )
            sd = additional_datasets[pt.constraint_str][bn]["safety_dataset"]
            assert sd.num_datapoints == int(
                round((frac_data_in_safety) * 0.8 * spec.dataset.num_datapoints)
            )

    # Run the Seldonian algorithm
    passed_safety, solution = SA.run()
    assert passed_safety == True
    array_to_compare = np.array(
        [
            4.18103862e-01,
            1.06776995e-04,
            8.46491836e-04,
            4.95734241e-04,
            5.21233786e-04,
            3.25287639e-04,
            2.10667062e-03,
            1.50025360e-03,
            -1.24865593e-04,
            4.89120616e-04,
        ]
    )
    assert np.allclose(solution, array_to_compare)


def test_gpa_data_classification(gpa_classification_dataset):
    """Test that the gpa classification example runs
    with the five fairness constraints (separately):
    Disparate impact
    Demographic parity
    Equalized odds
    Equal opportunity
    Predictive equality

    Make sure safety test passes and solution is correct.
    """
    rseed = 0
    np.random.seed(rseed)
    frac_data_in_safety = 0.6

    fairness_constraint_dict = {
        "disparate_impact": "0.8 - min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M]))",
        "demographic_parity": "abs((PR | [M]) - (PR | [F])) - 0.15",
        "equalized_odds": "abs((FNR | [M]) - (FNR | [F])) + abs((FPR | [M]) - (FPR | [F])) - 0.35",
        "equal_opportunity": "abs((FNR | [M]) - (FNR | [F])) - 0.2",
        "predictive_equality": "abs((FPR | [M]) - (FPR | [F])) - 0.2",
    }

    solution_dict = {
        "disparate_impact": np.array(
            [
                -0.14932756,
                -0.04743285,
                0.15603878,
                0.10953721,
                0.08014052,
                0.03997749,
                0.40484586,
                0.3045744,
                -0.1084586,
                -0.05770913,
            ]
        ),
        "demographic_parity": np.array(
            [
                -0.14932756,
                -0.04743285,
                0.15603878,
                0.10953721,
                0.08014052,
                0.03997749,
                0.40484586,
                0.3045744,
                -0.1084586,
                -0.05770913,
            ]
        ),
        "equalized_odds": np.array(
            [
                -0.14932756,
                -0.04743285,
                0.15603878,
                0.10953721,
                0.08014052,
                0.03997749,
                0.40484586,
                0.3045744,
                -0.1084586,
                -0.05770913,
            ]
        ),
        "equal_opportunity": np.array(
            [
                -0.14932756,
                -0.04743285,
                0.15603878,
                0.10953721,
                0.08014052,
                0.03997749,
                0.40484586,
                0.3045744,
                -0.1084586,
                -0.05770913,
            ]
        ),
        "predictive_equality": np.array(
            [
                -0.14932756,
                -0.04743285,
                0.15603878,
                0.10953721,
                0.08014052,
                0.03997749,
                0.40484586,
                0.3045744,
                -0.1084586,
                -0.05770913,
            ]
        ),
    }

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    for constraint in fairness_constraint_dict:
        print(constraint)
        constraint_str = fairness_constraint_dict[constraint]
        constraint_strs = [constraint_str]
        deltas = [0.05]

        (dataset, model, primary_objective, parse_trees) = gpa_classification_dataset(
            constraint_strs=constraint_strs, deltas=deltas
        )

        # Create spec object
        spec = SupervisedSpec(
            dataset=dataset,
            model=model,
            parse_trees=parse_trees,
            sub_regime="classification",
            frac_data_in_safety=frac_data_in_safety,
            primary_objective=primary_objective,
            use_builtin_primary_gradient_fn=False,
            initial_solution_fn=initial_solution_fn,
            optimization_technique="gradient_descent",
            optimizer="adam",
            optimization_hyperparams={
                "lambda_init": np.array([0.5]),
                "alpha_theta": 0.005,
                "alpha_lamb": 0.005,
                "beta_velocity": 0.9,
                "beta_rmsprop": 0.95,
                "num_iters": 10,
                "use_batches": False,
                "gradient_library": "autograd",
                "hyper_search": None,
                "verbose": True,
            },
        )

        # Run seldonian algorithm
        SA = SeldonianAlgorithm(spec)
        passed_safety, solution = SA.run()
        assert passed_safety == True
        print(solution)

        solution_to_compare = solution_dict[constraint]

        assert np.allclose(solution, solution_to_compare)


def test_gpa_data_classification_addl_datasets(gpa_classification_addl_datasets):
    """Test that the gpa classification example runs
    when using a different dataset for the base nodes compared to the primary objective
    """
    # Load metadata
    rseed = 0
    np.random.seed(rseed)
    constraint_strs = ["abs((PR | [M]) - (PR | [F])) - 0.15"]
    deltas = [0.05]

    (
        primary_dataset,
        additional_datasets,
        model,
        primary_objective,
        parse_trees,
    ) = gpa_classification_addl_datasets(constraint_strs=constraint_strs, deltas=deltas)

    frac_data_in_safety = 0.6

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    # Create spec object
    spec = SupervisedSpec(
        dataset=primary_dataset,
        additional_datasets=additional_datasets,
        model=model,
        parse_trees=parse_trees,
        sub_regime="classification",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 200,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Set up SA object
    SA = SeldonianAlgorithm(spec)
    # Ensure that the candidate and safety datasets were created within the additional_datasets object
    for pt in spec.parse_trees:
        base_nodes_this_tree = list(pt.base_node_dict.keys())
        for bn in base_nodes_this_tree:
            assert "candidate_dataset" in additional_datasets[pt.constraint_str][bn]
            assert "safety_dataset" in additional_datasets[pt.constraint_str][bn]
            bn_dataset = additional_datasets[pt.constraint_str][bn]["dataset"]
            assert bn_dataset.num_datapoints == int(
                round(spec.dataset.num_datapoints * 0.8)
            )
            cd = additional_datasets[pt.constraint_str][bn]["candidate_dataset"]
            assert cd.num_datapoints == int(
                round((1.0 - frac_data_in_safety) * 0.8 * spec.dataset.num_datapoints)
            )
            sd = additional_datasets[pt.constraint_str][bn]["safety_dataset"]
            assert sd.num_datapoints == int(
                round((frac_data_in_safety) * 0.8 * spec.dataset.num_datapoints)
            )

    # Run the Seldonian algorithm
    passed_safety, solution = SA.run()
    assert passed_safety == True
    array_to_compare = np.array(
        [
            -0.14932756,
            -0.04743285,
            0.15603878,
            0.10953721,
            0.08014052,
            0.03997749,
            0.40484586,
            0.3045744,
            -0.1084586,
            -0.05770913,
        ]
    )

    assert np.allclose(solution, array_to_compare)

    # Now use custom batch size for additional dataset. Provide custom batch size for one base node.
    # The other base node will get the batch size of the primary objective.
    # Need to turn on batching in spec object
    primary_batch_size = 5000
    batch_size_dict = {"abs((PR | [M]) - (PR | [F])) - 0.15": {"PR | [M]": 1000,}}
    (
        primary_dataset,
        additional_datasets,
        model,
        primary_objective,
        parse_trees,
    ) = gpa_classification_addl_datasets(
        constraint_strs=constraint_strs, deltas=deltas, batch_size_dict=batch_size_dict
    )

    # Create new spec object
    spec2 = SupervisedSpec(
        dataset=primary_dataset,
        additional_datasets=additional_datasets,
        model=model,
        parse_trees=parse_trees,
        sub_regime="classification",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "use_batches": True,
            "n_epochs": 5,
            "batch_size": primary_batch_size,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Set up SA object
    SA2 = SeldonianAlgorithm(spec2)
    # Ensure that the candidate and safety datasets were created within the additional_datasets object
    for pt in spec2.parse_trees:
        base_nodes_this_tree = list(pt.base_node_dict.keys())
        for bn in base_nodes_this_tree:
            assert "candidate_dataset" in additional_datasets[pt.constraint_str][bn]
            assert "safety_dataset" in additional_datasets[pt.constraint_str][bn]
            if "batch_size" in additional_datasets[pt.constraint_str][bn]:
                assert (
                    additional_datasets[pt.constraint_str][bn]["batch_size"]
                    == batch_size_dict[pt.constraint_str][bn]
                )
            cd = additional_datasets[pt.constraint_str][bn]["candidate_dataset"]
            assert cd.num_datapoints == int(
                round((1.0 - frac_data_in_safety) * 0.8 * spec.dataset.num_datapoints)
            )
            sd = additional_datasets[pt.constraint_str][bn]["safety_dataset"]
            assert sd.num_datapoints == int(
                round((frac_data_in_safety) * 0.8 * spec.dataset.num_datapoints)
            )

    # Run the Seldonian algorithm
    passed_safety, solution = SA2.run()
    assert passed_safety == True
    array_to_compare = np.array(
        [
            -0.13806723,
            -0.0342689,
            0.16904019,
            0.12271661,
            0.09261192,
            0.05334552,
            0.39152464,
            0.291386,
            -0.09522752,
            -0.04463774,
        ]
    )

    assert np.allclose(solution, array_to_compare)


def test_classification_statistics(gpa_classification_dataset):
    """Test all of the classification statistics (FPR, PR, NR, etc.)
    are evaluated properly for the GPA dataset
    where we know what the answers should be

    """
    rseed = 0
    np.random.seed(rseed)
    frac_data_in_safety = 0.6

    constraint_str = "(PR + NR + FPR + FNR + TPR + TNR) - 10.0"
    constraint_strs = [constraint_str]
    deltas = [0.05]

    (dataset, model, primary_objective, parse_trees) = gpa_classification_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    # Create spec object
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="classification",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 25,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Run seldonian algorithm
    SA = SeldonianAlgorithm(spec)
    passed_safety, solution = SA.run()
    assert passed_safety == True
    print(passed_safety, solution)
    solution_to_compare = np.array(
        [
            -0.14932756,
            -0.04743285,
            0.15603878,
            0.10953721,
            0.08014052,
            0.03997749,
            0.40484586,
            0.3045744,
            -0.1084586,
            -0.05770913,
        ]
    )

    assert np.allclose(solution, solution_to_compare)


def test_NSF(gpa_regression_dataset):
    """Test that no solution is found for a constraint
    that is impossible to satisfy, e.g. negative mean squared error.
    Make sure that candidate selection did return a solution though
    """
    rseed = 0
    np.random.seed(rseed)
    constraint_strs = ["Mean_Squared_Error + 2.0"]
    deltas = [0.05]

    (dataset, model, primary_objective, parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )

    frac_data_in_safety = 0.6

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    # Create spec object
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 100,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Run seldonian algorithm
    SA = SeldonianAlgorithm(spec)
    passed_safety, solution = SA.run()
    print(f"Solution: {solution}")
    assert passed_safety == False
    assert solution == "NSF"

    res = SA.get_cs_result()
    candidate_solution = res["candidate_solution"]
    assert isinstance(candidate_solution, np.ndarray)

    # Test that evaluate primary objective function raises a value error
    with pytest.raises(ValueError) as excinfo:
        SA.evaluate_primary_objective(branch="candidate_solution", theta=solution)

    assert str(excinfo.value) == "Cannot evaluate primary objective because theta='NSF'"

    # Test that evaluate primary objective function raises a value error
    with pytest.raises(ValueError) as excinfo:
        SA.evaluate_primary_objective(branch="safety_test", theta=solution)

    assert str(excinfo.value) == "Cannot evaluate primary objective because theta='NSF'"


def test_cmaes(gpa_regression_dataset):
    """Test that the CMA-ES black box optimizers successfully optimize the GPA
    regression problem with a simple non-conflicting constraint
    """
    rseed = 99
    np.random.seed(rseed)
    constraint_strs = ["Mean_Squared_Error - 2.0"]
    deltas = [0.05]
    (dataset, model, primary_objective, parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )

    frac_data_in_safety = 0.6

    array_to_compare = np.array(
        [
            4.17882264e-01,
            -1.59868384e-04,
            6.33766780e-04,
            2.64271363e-04,
            3.08303718e-04,
            1.01170148e-04,
            1.86987938e-03,
            1.29098726e-03,
            -3.82405534e-04,
            2.29938169e-04,
        ]
    )

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    # for optimizer in ['Powell','CG','Nelder-Mead','BFGS','CMA-ES']:
    for optimizer in ["CMA-ES"]:
        spec = SupervisedSpec(
            dataset=dataset,
            model=model,
            parse_trees=parse_trees,
            sub_regime="regression",
            frac_data_in_safety=frac_data_in_safety,
            primary_objective=primary_objective,
            use_builtin_primary_gradient_fn=False,
            initial_solution_fn=initial_solution_fn,
            optimization_technique="barrier_function",
            optimizer=optimizer,
            optimization_hyperparams={
                "maxiter": 100 if optimizer == "CMA-ES" else 1000,
                "seed": rseed,
                "hyper_search": None,
                "verbose": True,
            },
        )

        # Run seldonian algorithm
        SA = SeldonianAlgorithm(spec)
        passed_safety, solution = SA.run()

        assert passed_safety == True
        if optimizer != "CMA-ES":
            # CMA-ES might come up with a different solution on test server
            assert np.allclose(solution, array_to_compare)


def test_use_custom_primary_gradient(gpa_regression_dataset):
    """Test that the gpa regression example runs
    when using a custom primary gradient function.
    It is the same as the built-in but passed as
    a custom function. Make
    sure safety test passes and solution is correct.
    """

    rseed = 0
    np.random.seed(rseed)
    constraint_strs = ["Mean_Squared_Error - 2.0"]
    deltas = [0.05]

    (dataset, model, primary_objective, parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )

    frac_data_in_safety = 0.6

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    # Create spec object
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=False,
        custom_primary_gradient_fn=objectives.gradient_Mean_Squared_Error,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 200,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Run seldonian algorithm
    SA = SeldonianAlgorithm(spec)
    passed_safety, solution = SA.run()
    assert passed_safety == True
    array_to_compare = np.array(
        [
            4.17882259e-01,
            -1.59868384e-04,
            6.33766780e-04,
            2.64271363e-04,
            3.08303718e-04,
            1.01170148e-04,
            1.86987938e-03,
            1.29098727e-03,
            -3.82405534e-04,
            2.29938169e-04,
        ]
    )
    assert np.allclose(solution, array_to_compare)


def test_get_candidate_selection_result(gpa_regression_dataset):
    """Test that the after running the SA on the
    gpa regression example, we can get the
    full candidate selection solution dictionary
    from gradient descent as a method call on the
    SA() object.

    Also check that before we run SA.run() this same
    method gives us an error.
    """

    rseed = 0
    np.random.seed(rseed)
    constraint_strs = ["Mean_Squared_Error - 2.0"]
    deltas = [0.05]

    (dataset, model, primary_objective, parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )

    frac_data_in_safety = 0.6

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    # Create spec object
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 100,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # # Run seldonian algorithm
    SA = SeldonianAlgorithm(spec)
    # Try to get candidate solution result before running
    with pytest.raises(ValueError) as excinfo:
        res = SA.get_cs_result()
    error_str = "Candidate selection has not been run yet, so result is not available."
    assert error_str in str(excinfo.value)

    passed_safety, solution = SA.run()
    res = SA.get_cs_result()
    res_keys = res.keys()
    for key in [
        "candidate_solution",
        "best_index",
        "best_g",
        "best_f",
        "f_vals",
        "g_vals",
        "lamb_vals",
        "L_vals",
    ]:
        assert key in res_keys


def test_get_safety_test_result(gpa_regression_dataset):
    """Test that the after running the SA on the
    gpa regression example, we can get the
    dictionary containing the parse trees evaluated
    on the safety test. We also test the method
    that retrieves the upper bounds.

    Also check that before we run SA.run() this same
    method gives us an error.
    """

    rseed = 0
    np.random.seed(rseed)
    constraint_strs = ["Mean_Squared_Error >= 1.25", "Mean_Squared_Error <= 2.0"]
    deltas = [0.1, 0.1]

    (dataset, model, primary_objective, parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )

    frac_data_in_safety = 0.6

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    # Create spec object
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": 0.5,
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 150,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # # Run seldonian algorithm
    SA = SeldonianAlgorithm(spec)
    # Try to get candidate solution result before running
    with pytest.raises(ValueError) as excinfo:
        res = SA.get_st_upper_bounds()
    error_str = "Safety test has not been run yet, so upper bounds are not available."
    assert error_str in str(excinfo.value)

    passed_safety, solution = SA.run()
    assert passed_safety == True
    res = SA.get_st_upper_bounds()
    assert len(res) == 2
    print(res)
    assert res["1.25-(Mean_Squared_Error)"] == pytest.approx(-0.19604227384297923)
    assert res["Mean_Squared_Error-(2.0)"] == pytest.approx(-0.5219448029759275)


def test_nans_infs_gradient_descent(gpa_regression_dataset):
    """Test that if nans or infs appear in theta in gradient
    descent then the algorithm returns whatever the best solution
    has been so far.
    """
    rseed = 0
    np.random.seed(rseed)
    constraint_strs = ["Mean_Squared_Error - 2.0"]
    deltas = [0.05]

    (dataset, model, primary_objective, parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )

    frac_data_in_safety = 0.6
    # first nans
    initial_solution_fn_nan = lambda m, x, y: np.nan * np.ones(10)
    # Create spec object
    spec_nan = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn_nan,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 200,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Run seldonian algorithm
    SA_nan = SeldonianAlgorithm(spec_nan)
    passed_safety_nan, solution_nan = SA_nan.run(debug=True)
    assert passed_safety_nan == False
    assert solution_nan == "NSF"

    # now infs
    initial_solution_fn_inf = lambda m, x, y: np.inf * np.ones(10)
    # Create spec object
    spec_inf = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn_inf,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 200,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Run seldonian algorithm
    SA_inf = SeldonianAlgorithm(spec_inf)
    passed_safety_inf, solution_inf = SA_inf.run(debug=True)
    assert passed_safety_inf == False
    assert solution_inf == "NSF"


def test_run_safety_test_only(gpa_regression_dataset):
    """Test that the after running the SA on the
    gpa regression example, we can get the
    full candidate selection solution dictionary
    from gradient descent as a method call on the
    SA() object.

    Also check that before we run SA.run() this same
    method gives us an error.
    """

    rseed = 0
    np.random.seed(rseed)
    constraint_strs = ["Mean_Squared_Error - 2.0"]
    deltas = [0.05]

    (dataset, model, primary_objective, parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )

    frac_data_in_safety = 0.6

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    # Create spec object
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 100,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # # Run seldonian algorithm
    SA = SeldonianAlgorithm(spec)
    # Try to get candidate solution result before running
    test_solution = np.array(
        [
            4.17882259e-01,
            -1.59868384e-04,
            6.33766780e-04,
            2.64271363e-04,
            3.08303718e-04,
            1.01170148e-04,
            1.86987938e-03,
            1.29098727e-03,
            -3.82405534e-04,
            2.29938169e-04,
        ]
    )
    passed_safety, solution = SA.run_safety_test(test_solution)
    assert passed_safety == True
    assert np.allclose(test_solution, solution)


def test_reg_coef(gpa_regression_dataset):
    """Test that using a regularization coefficient
    works
    """

    rseed = 0
    np.random.seed(rseed)
    constraint_strs = ["Mean_Squared_Error - 2.0"]
    deltas = [0.05]

    (dataset, model, primary_objective, parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )

    frac_data_in_safety = 0.6

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    # First gradient descent
    # Create spec object
    spec_gs = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 100,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
        regularization_hyperparams={"reg_coef": 0.5},
    )

    # # Run seldonian algorithm
    SA_gs = SeldonianAlgorithm(spec_gs)
    # Try to get candidate solution result before running
    test_solution_gs = np.array(
        [
            4.17882259e-01,
            -1.59868384e-04,
            6.33766780e-04,
            2.64271363e-04,
            3.08303718e-04,
            1.01170148e-04,
            1.86987938e-03,
            1.29098727e-03,
            -3.82405534e-04,
            2.29938169e-04,
        ]
    )
    passed_safety, solution = SA_gs.run()
    assert passed_safety == True
    assert np.allclose(test_solution_gs, solution)

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    spec_bb = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="barrier_function",
        optimizer="Powell",
        regularization_hyperparams={"reg_coef": 0.5},
    )

    # # Run seldonian algorithm
    SA_bb = SeldonianAlgorithm(spec_bb)
    # Try to get candidate solution result before running
    test_solution_bb = np.array(
        [
            1.26219949e-04,
            3.59203006e-04,
            9.26674215e-04,
            4.18683641e-04,
            3.62709523e-04,
            3.48171863e-05,
            1.90106843e-03,
            1.31441205e-03,
            -6.56374856e-04,
            2.12829138e-04,
        ]
    )
    passed_safety, solution = SA_bb.run(debug=True)
    assert passed_safety == True
    assert np.allclose(test_solution_bb, solution)


def test_create_logfile(gpa_regression_dataset):
    """Test that using a regularization coefficient
    works
    """
    # Check how many logs there are before test:
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    logfiles_before = os.listdir(log_dir)
    n_before = len(logfiles_before)
    rseed = 0
    np.random.seed(rseed)
    constraint_strs = ["Mean_Squared_Error - 2.0"]
    deltas = [0.05]

    (dataset, model, primary_objective, parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )

    frac_data_in_safety = 0.6

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    # First gradient descent
    # Create spec object
    spec_gs = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 2,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # # Run seldonian algorithm
    SA_gs = SeldonianAlgorithm(spec_gs)
    # Try to get candidate solution result before running
    passed_safety, solution = SA_gs.run(write_cs_logfile=True)
    logfiles_after = os.listdir(log_dir)
    n_after = len(logfiles_after)
    assert n_after == n_before + 1


def test_bad_autodiff_method(gpa_classification_dataset):
    """Test that using a regularization coefficient
    works
    """
    constraint_str = "PR >= 0.9"
    constraint_strs = [constraint_str]
    deltas = [0.05]

    (dataset, model, primary_objective, parse_trees) = gpa_classification_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )
    frac_data_in_safety = 0.6

    # Create spec object
    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="classification",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 25,
            "use_batches": False,
            "gradient_library": "superfast",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Run seldonian algorithm
    SA = SeldonianAlgorithm(spec)
    with pytest.raises(NotImplementedError) as excinfo:
        passed_safety, solution = SA.run()

    error_str = "gradient library: superfast not supported"

    assert str(excinfo.value) == error_str


def test_lambda_init(gpa_regression_dataset):
    """Test that lambda given with correct shape
    works but with wrong shape raises an error
    """
    # Load metadata
    rseed = 0
    np.random.seed(rseed)
    constraint_strs = ["Mean_Squared_Error - 5.0", "2.0 - Mean_Squared_Error"]
    deltas = [0.05, 0.1]

    (dataset, model, primary_objective, parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )

    frac_data_in_safety = 0.6

    # A float can work - assumption is that all constraints get this value
    hyperparams1 = {
        "lambda_init": 0.5,
        "alpha_theta": 0.005,
        "alpha_lamb": 0.005,
        "beta_velocity": 0.9,
        "beta_rmsprop": 0.95,
        "num_iters": 2,
        "use_batches": False,
        "gradient_library": "autograd",
        "hyper_search": None,
        "verbose": True,
    }

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    # Create spec object
    spec1 = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams=hyperparams1,
    )

    # Run seldonian algorithm
    SA1 = SeldonianAlgorithm(spec1)
    passed_safety, solution = SA1.run()

    hyperparams2 = {
        "lambda_init": np.array([0.5, 0.25]),
        "alpha_theta": 0.005,
        "alpha_lamb": 0.005,
        "beta_velocity": 0.9,
        "beta_rmsprop": 0.95,
        "num_iters": 2,
        "use_batches": False,
        "gradient_library": "autograd",
        "hyper_search": None,
        "verbose": True,
    }

    # Create spec object
    spec2 = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams=hyperparams2,
    )

    # Run seldonian algorithm
    SA2 = SeldonianAlgorithm(spec2)
    passed_safety, solution = SA2.run()

    hyperparams3 = {
        "lambda_init": np.array([0.5]),
        "alpha_theta": 0.005,
        "alpha_lamb": 0.005,
        "beta_velocity": 0.9,
        "beta_rmsprop": 0.95,
        "num_iters": 2,
        "use_batches": False,
        "gradient_library": "autograd",
        "hyper_search": None,
        "verbose": True,
    }
    # Create spec object
    spec3 = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams=hyperparams3,
    )

    # Run seldonian algorithm
    SA3 = SeldonianAlgorithm(spec3)
    with pytest.raises(RuntimeError) as excinfo:
        passed_safety, solution = SA3.run()

    error_str = (
        "lambda has wrong shape. "
        "Shape must be (n_constraints,), "
        "but shape is (1,)"
    )
    assert str(excinfo.value) == error_str

    # Allow a list to be passed
    hyperparams4 = {
        "lambda_init": [0.05, 0.15],
        "alpha_theta": 0.005,
        "alpha_lamb": 0.005,
        "beta_velocity": 0.9,
        "beta_rmsprop": 0.95,
        "num_iters": 2,
        "use_batches": False,
        "gradient_library": "autograd",
        "hyper_search": None,
        "verbose": True,
    }
    # Create spec object
    spec4 = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams=hyperparams4,
    )

    # Run seldonian algorithm
    SA4 = SeldonianAlgorithm(spec4)

    passed_safety, solution = SA4.run()

    # But not a list of the wrong length
    hyperparams5 = {
        "lambda_init": [0.05],
        "alpha_theta": 0.005,
        "alpha_lamb": 0.005,
        "beta_velocity": 0.9,
        "beta_rmsprop": 0.95,
        "num_iters": 2,
        "use_batches": False,
        "gradient_library": "autograd",
        "hyper_search": None,
        "verbose": True,
    }
    # Create spec object
    spec5 = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams=hyperparams5,
    )

    # Run seldonian algorithm
    SA5 = SeldonianAlgorithm(spec5)

    with pytest.raises(RuntimeError) as excinfo:
        passed_safety, solution = SA5.run()

    error_str = (
        "lambda has wrong shape. "
        "Shape must be (n_constraints,), "
        "but shape is (1,)"
    )
    assert str(excinfo.value) == error_str

    # Don't allow an array that has too many dimensions
    hyperparams6 = {
        "lambda_init": np.array([[0.05]]),
        "alpha_theta": 0.005,
        "alpha_lamb": 0.005,
        "beta_velocity": 0.9,
        "beta_rmsprop": 0.95,
        "num_iters": 2,
        "use_batches": False,
        "gradient_library": "autograd",
        "hyper_search": None,
        "verbose": True,
    }
    # Create spec object
    spec6 = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams=hyperparams6,
    )

    # Run seldonian algorithm
    SA6 = SeldonianAlgorithm(spec6)

    with pytest.raises(RuntimeError) as excinfo:
        passed_safety, solution = SA6.run()

    error_str = (
        "lambda has wrong shape. "
        "Shape must be (n_constraints,), "
        "but shape is (1, 1)"
    )
    assert str(excinfo.value) == error_str


def test_no_primary_provided(
    gpa_regression_dataset, gpa_classification_dataset, RL_gridworld_dataset
):
    """Test that if the user does not provide a primary objective,
    then the default is used in the three different regimes/sub-regimes
    """
    # Regression
    rseed = 99
    np.random.seed(rseed)
    constraint_strs = ["Mean_Squared_Error - 2.0"]
    deltas = [0.05]
    (dataset, model, _, parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )
    frac_data_in_safety = 0.6

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=None,
        use_builtin_primary_gradient_fn=False,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": 0.5,
            "alpha_theta": 0.01,
            "alpha_lamb": 0.01,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 2,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )
    assert spec.primary_objective == None

    # Create seldonian algorithm object, which assigns primary objective
    SA = SeldonianAlgorithm(spec)
    assert spec.primary_objective != None
    assert spec.primary_objective.__name__ == "Mean_Squared_Error"

    # Classification
    constraint_strs = ["FPR - 0.5"]
    deltas = [0.05]

    (dataset, model, _, parse_trees) = gpa_classification_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )

    # Create spec object

    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="classification",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=None,
        use_builtin_primary_gradient_fn=False,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 10,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )
    assert spec.primary_objective == None

    # Create seldonian algorithm object, which assigns primary objective
    SA = SeldonianAlgorithm(spec)
    assert spec.primary_objective != None
    assert spec.primary_objective.__name__ == "binary_logistic_loss"

    # RL
    constraint_strs = ["-0.25 - J_pi_new_IS"]
    deltas = [0.05]

    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,
        deltas,
        regime="reinforcement_learning",
        sub_regime="all",
        delta_weight_method="equal",
    )
    (dataset, policy, env_kwargs, _) = RL_gridworld_dataset()

    frac_data_in_safety = 0.6

    # Model

    model = RL_model(policy=policy, env_kwargs=env_kwargs)

    # Create spec object
    spec = RLSpec(
        dataset=dataset,
        model=model,
        frac_data_in_safety=frac_data_in_safety,
        use_builtin_primary_gradient_fn=True,
        primary_objective=None,
        parse_trees=parse_trees,
        initial_solution_fn=None,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": 0.5,
            "alpha_theta": 0.01,
            "alpha_lamb": 0.01,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 2,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )
    assert spec.primary_objective == None

    # Create seldonian algorithm object, which assigns primary objective
    SA = SeldonianAlgorithm(spec)
    assert spec.primary_objective != None
    assert spec.primary_objective.__name__ == "IS_estimate"


def test_no_initial_solution_provided(
    gpa_regression_dataset,
    gpa_classification_dataset,
    gpa_multiclass_dataset,
    RL_gridworld_dataset,
):
    """Test that if the user does not provide a primary objective,
    then the default is used in the three different regimes/sub-regimes
    """
    # Regression
    rseed = 99
    np.random.seed(rseed)
    constraint_strs = ["Mean_Squared_Error - 2.0"]
    deltas = [0.05]
    (dataset, model, primary_objective, parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )
    frac_data_in_safety = 0.6

    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="regression",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=False,
        initial_solution_fn=None,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": 0.5,
            "alpha_theta": 0.01,
            "alpha_lamb": 0.01,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 2,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )
    SA = SeldonianAlgorithm(spec)
    SA.set_initial_solution()
    assert np.allclose(SA.initial_solution, np.zeros(10))

    # Binary Classification
    constraint_strs = ["FPR - 0.5"]
    deltas = [0.05]

    (dataset, model, primary_objective, parse_trees) = gpa_classification_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )

    # Create spec object

    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="binary_classification",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=False,
        initial_solution_fn=None,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 2,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Create seldonian algorithm object
    SA = SeldonianAlgorithm(spec)
    SA.set_initial_solution()
    assert np.allclose(SA.initial_solution, np.zeros(10))

    # Multi-class Classification
    constraint_strs = ["CM_[0,0] >= 0.25"]
    deltas = [0.05]

    (dataset, model, primary_objective, parse_trees) = gpa_multiclass_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )

    # Create spec object

    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime="multiclass_classification",
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=False,
        initial_solution_fn=None,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 2,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Create seldonian algorithm object
    SA = SeldonianAlgorithm(spec)
    SA.set_initial_solution()
    assert np.allclose(SA.initial_solution, np.zeros((10, 3)))


def test_gpa_decision_tree(gpa_classification_dataset):
    """Test that the decision tree model works
    with the gpa classification example with disparate impact

    Make sure safety test passes and solution is correct.
    """
    from seldonian.models.trees.sktree_model import SeldonianDecisionTree, probs2theta

    rseed = 0
    np.random.seed(rseed)
    frac_data_in_safety = 0.6
    model = SeldonianDecisionTree(max_depth=4)

    fairness_constraint_dict = {
        "disparate_impact": "0.8 - min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M]))",
    }

    solution_dict = {
        "disparate_impact": np.array(
            [
                -0.91629073,
                -0.3438721,
                -1.09239933,
                -0.66233601,
                -0.52598787,
                0.39204209,
                -0.03608235,
                0.89994159,
                -0.0151407,
                1.02850272,
                0.1308396,
                1.26640574,
                0.45630478,
                1.17799732,
                -0.87546874,
                2.004718,
            ]
        ),
    }

    def initial_solution_fn(m, x, y):
        probs = m.fit(x, y)
        return probs2theta(probs)

    for constraint in fairness_constraint_dict:
        constraint_str = fairness_constraint_dict[constraint]
        constraint_strs = [constraint_str]
        deltas = [0.05]

        (dataset, _, primary_objective, parse_trees) = gpa_classification_dataset(
            constraint_strs=constraint_strs, deltas=deltas
        )

        # Create spec object
        spec = SupervisedSpec(
            dataset=dataset,
            model=model,
            parse_trees=parse_trees,
            sub_regime="classification",
            frac_data_in_safety=frac_data_in_safety,
            primary_objective=primary_objective,
            use_builtin_primary_gradient_fn=False,
            initial_solution_fn=initial_solution_fn,
            optimization_technique="gradient_descent",
            optimizer="adam",
            optimization_hyperparams={
                "lambda_init": np.array([0.5]),
                "alpha_theta": 0.005,
                "alpha_lamb": 0.005,
                "beta_velocity": 0.9,
                "beta_rmsprop": 0.95,
                "num_iters": 10,
                "use_batches": False,
                "gradient_library": "autograd",
                "hyper_search": None,
                "verbose": True,
            },
        )

        # Run seldonian algorithm
        SA = SeldonianAlgorithm(spec)
        passed_safety, solution = SA.run()
        assert passed_safety == True

        solution_to_compare = solution_dict[constraint]

        assert np.allclose(solution, solution_to_compare)


def test_gpa_random_forest(gpa_classification_dataset):
    """Test that the decision tree model works
    with the gpa classification example with disparate impact

    Make sure safety test passes and solution is correct.
    """
    from seldonian.models.trees.skrandomforest_model import (
        SeldonianRandomForest,
        probs2theta,
    )

    rseed = 0
    np.random.seed(rseed)
    frac_data_in_safety = 0.6
    model = SeldonianRandomForest(max_depth=3, n_estimators=5)

    fairness_constraint_dict = {
        "disparate_impact": "0.8 - min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M]))",
    }

    solution_dict = {
        "disparate_impact": np.array(
            [
                -0.80923242,
                -0.24296163,
                -0.32121083,
                0.58102988,
                0.20254449,
                1.43074612,
                0.99403948,
                2.49842037,
                -0.42527665,
                -0.85068235,
                -0.00595723,
                0.82507472,
                -0.0133283,
                0.94150215,
                0.7472144,
                2.52572864,
                -0.42381425,
                -0.91412623,
                -0.408637,
                0.21814065,
                -0.15094095,
                0.64132211,
                0.75391354,
                2.26610688,
                -0.79259672,
                -0.19415601,
                -0.2897921,
                0.7021469,
                -0.22314355,
                0.68145114,
                0.94243012,
                2.18287713,
                -0.22875438,
                -0.67453988,
                -0.31015493,
                0.40171276,
                -0.1057952,
                0.35916793,
                0.66489671,
                1.88178562,
            ]
        ),
    }

    def initial_solution_fn(m, x, y):
        probs = m.fit(x, y)
        return probs2theta(probs)

    for constraint in fairness_constraint_dict:
        constraint_str = fairness_constraint_dict[constraint]
        constraint_strs = [constraint_str]
        deltas = [0.05]

        (dataset, _, primary_objective, parse_trees) = gpa_classification_dataset(
            constraint_strs=constraint_strs, deltas=deltas
        )

        # Create spec object
        spec = SupervisedSpec(
            dataset=dataset,
            model=model,
            parse_trees=parse_trees,
            sub_regime="classification",
            frac_data_in_safety=frac_data_in_safety,
            primary_objective=primary_objective,
            use_builtin_primary_gradient_fn=False,
            initial_solution_fn=initial_solution_fn,
            optimization_technique="gradient_descent",
            optimizer="adam",
            optimization_hyperparams={
                "lambda_init": np.array([0.5]),
                "alpha_theta": 0.005,
                "alpha_lamb": 0.005,
                "beta_velocity": 0.9,
                "beta_rmsprop": 0.95,
                "num_iters": 10,
                "use_batches": False,
                "gradient_library": "autograd",
                "hyper_search": None,
                "verbose": True,
            },
        )

        # Run seldonian algorithm
        SA = SeldonianAlgorithm(spec)
        passed_safety, solution = SA.run()
        assert passed_safety == True

        solution_to_compare = solution_dict[constraint]
        assert np.allclose(solution, solution_to_compare)


""" Custom regime tests """


def test_custom_text_dataset(custom_text_spec):
    # Test that the custom dataset (lists of strings) runs all the way through the algorithm
    np.random.seed(0)
    spec = custom_text_spec()
    SA = SeldonianAlgorithm(spec)
    passed_safety, solution = SA.run()
    expected = np.array([-1.01, -0.01, 0.99])
    assert passed_safety == True
    assert np.allclose(solution, expected)


def test_custom_loan_dataset(custom_loan_spec):
    # Test that the loan dataset with regime="custom" runs all the way through the algorithm
    # This tests using conditional columns with the custom regime, which the custom text dataset does not
    # Also tests batching
    np.random.seed(0)
    spec = custom_loan_spec()
    SA = SeldonianAlgorithm(spec)
    passed_safety, solution = SA.run(debug=True)

    assert len(solution) == 58
    assert passed_safety == True


def test_custom_loan_addl_dataset(custom_loan_addl_dataset):
    # Test that the loan dataset with regime="custom" runs all the way through the algorithm
    # This tests using conditional columns with the custom regime, which the custom text dataset does not

    rseed = 0
    np.random.seed(rseed)

    (
        primary_dataset,
        additional_datasets,
        model,
        primary_objective,
        parse_trees,
    ) = custom_loan_addl_dataset()

    frac_data_in_safety = 0.6

    def custom_initial_solution_fn(model, data, **kwargs):
        features = data[:, :-1]
        labels = data[:, -1]
        return model.fit(features, labels)

    # Create spec object
    spec = Spec(
        dataset=primary_dataset,
        additional_datasets=additional_datasets,
        model=model,
        parse_trees=parse_trees,
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=False,
        initial_solution_fn=custom_initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.005,
            "alpha_lamb": 0.005,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 200,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    SA = SeldonianAlgorithm(spec)

    # Ensure that the candidate and safety datasets were created within the additional_datasets object
    for pt in spec.parse_trees:
        base_nodes_this_tree = list(pt.base_node_dict.keys())
        for bn in base_nodes_this_tree:
            assert "candidate_dataset" in additional_datasets[pt.constraint_str][bn]
            assert "safety_dataset" in additional_datasets[pt.constraint_str][bn]
            bn_dataset = additional_datasets[pt.constraint_str][bn]["dataset"]
            assert bn_dataset.num_datapoints == 500
            cd = additional_datasets[pt.constraint_str][bn]["candidate_dataset"]
            assert cd.num_datapoints == int(round((1.0 - frac_data_in_safety) * 500))
            sd = additional_datasets[pt.constraint_str][bn]["safety_dataset"]
            assert sd.num_datapoints == int(round((frac_data_in_safety) * 500))

    # Run the Seldonian algorithm
    passed_safety, solution = SA.run()
    assert passed_safety == True
    assert len(solution) == 58


""" RL based tests """


def test_RL_builtin_or_custom_gradient_not_supported(RL_gridworld_dataset):
    """Test that an error is raised if user tries to
    use built-in gradient or a custom gradient
    when doing RL
    """
    rseed = 99
    np.random.seed(rseed)
    constraint_strs = ["-0.25 - J_pi_new_IS"]
    deltas = [0.05]

    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,
        deltas,
        regime="reinforcement_learning",
        sub_regime="all",
        columns=[],
        delta_weight_method="equal",
    )
    (dataset, policy, env_kwargs, primary_objective) = RL_gridworld_dataset()

    frac_data_in_safety = 0.6

    # Model

    model = RL_model(policy=policy, env_kwargs=env_kwargs)

    # Create spec object
    spec = RLSpec(
        dataset=dataset,
        model=model,
        frac_data_in_safety=frac_data_in_safety,
        use_builtin_primary_gradient_fn=True,
        primary_objective=primary_objective,
        parse_trees=parse_trees,
        initial_solution_fn=None,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": 0.5,
            "alpha_theta": 0.01,
            "alpha_lamb": 0.01,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 2,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Run seldonian algorithm, making sure we capture error
    error_str = (
        "Using a builtin primary objective gradient"
        " is not yet supported for regimes other"
        " than supervised learning"
    )
    with pytest.raises(NotImplementedError) as excinfo:
        SA = SeldonianAlgorithm(spec)
        passed_safety, solution = SA.run()

    assert error_str in str(excinfo.value)

    # # # Create spec object
    spec2 = RLSpec(
        dataset=dataset,
        model=model,
        frac_data_in_safety=frac_data_in_safety,
        use_builtin_primary_gradient_fn=False,
        custom_primary_gradient_fn=lambda x: x,
        primary_objective=primary_objective,
        parse_trees=parse_trees,
        initial_solution_fn=None,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": 0.5,
            "alpha_theta": 0.01,
            "alpha_lamb": 0.01,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 2,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Run seldonian algorithm, making sure we capture error
    error_str2 = (
        "Using a provided primary objective gradient "
        "is not yet supported for regime='reinforcement_learning'."
    )
    with pytest.raises(NotImplementedError) as excinfo2:
        SA = SeldonianAlgorithm(spec2)
        passed_safety, solution = SA.run()

    assert error_str2 == str(excinfo2.value)


def test_RL_gridworld_gradient_descent(RL_gridworld_dataset):
    """Test that the RL gridworld example runs
    with a simple performance improvement constraint. Make
    sure safety test passes and solution is correct.
    """

    # IS estimate
    # Load data and metadata
    rseed = 99
    np.random.seed(rseed)
    constraint_strs = ["-10.0 - J_pi_new_IS"]
    deltas = [0.05]

    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,
        deltas,
        regime="reinforcement_learning",
        sub_regime="all",
        columns=[],
        delta_weight_method="equal",
    )
    (dataset, policy, env_kwargs, primary_objective) = RL_gridworld_dataset()

    frac_data_in_safety = 0.6
    model = RL_model(policy=policy, env_kwargs=env_kwargs)
    # Create spec object
    spec = RLSpec(
        dataset=dataset,
        model=model,
        frac_data_in_safety=frac_data_in_safety,
        use_builtin_primary_gradient_fn=False,
        primary_objective=primary_objective,
        parse_trees=parse_trees,
        initial_solution_fn=None,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": 0.5,
            "alpha_theta": 0.01,
            "alpha_lamb": 0.01,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 5,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # # Run seldonian algorithm
    SA = SeldonianAlgorithm(spec)
    passed_safety, solution = SA.run()
    assert passed_safety == True
    g_vals = SA.cs_result["g_vals"]
    assert g_vals[1][0] == pytest.approx(-9.67469087)

    # Get primary objective
    primary_val_st = SA.evaluate_primary_objective(theta=solution, branch="safety_test")
    assert primary_val_st == pytest.approx(0.42407173678433796)

    # WIS estimate
    # Load data and metadata
    rseed = 99
    np.random.seed(rseed)
    constraint_strs = ["-10.0 - J_pi_new_WIS"]
    deltas = [0.05]

    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,
        deltas,
        regime="reinforcement_learning",
        sub_regime="all",
        columns=[],
        delta_weight_method="equal",
    )
    (dataset, policy, env_kwargs, _) = RL_gridworld_dataset()

    frac_data_in_safety = 0.6
    model = RL_model(policy=policy, env_kwargs=env_kwargs)
    # Create spec object
    spec = RLSpec(
        dataset=dataset,
        model=model,
        frac_data_in_safety=frac_data_in_safety,
        use_builtin_primary_gradient_fn=False,
        primary_objective=objectives.WIS_estimate,
        parse_trees=parse_trees,
        initial_solution_fn=None,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": 0.5,
            "alpha_theta": 0.01,
            "alpha_lamb": 0.01,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 5,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # # Run seldonian algorithm
    SA = SeldonianAlgorithm(spec)
    passed_safety, solution = SA.run()
    assert passed_safety == True
    g_vals = SA.cs_result["g_vals"]
    assert g_vals[1][0] == pytest.approx(-9.67416473)

    # Get primary objective
    primary_val_st = SA.evaluate_primary_objective(theta=solution, branch="safety_test")
    assert primary_val_st == pytest.approx(0.43915764584186856)


def test_RL_gridworld_black_box(RL_gridworld_dataset):
    """Test that trying to run RL example with
    black box optimization gives a NotImplementedError,
    because it is not yet supported
    """
    # Load data and metadata
    rseed = 99
    np.random.seed(rseed)
    constraint_strs = ["-0.25 - J_pi_new_IS"]
    deltas = [0.05]

    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,
        deltas,
        regime="reinforcement_learning",
        sub_regime="all",
        columns=[],
        delta_weight_method="equal",
    )
    (dataset, policy, env_kwargs, primary_objective) = RL_gridworld_dataset()

    frac_data_in_safety = 0.6
    model = RL_model(policy=policy, env_kwargs=env_kwargs)
    # Create spec object
    spec = RLSpec(
        dataset=dataset,
        model=model,
        frac_data_in_safety=frac_data_in_safety,
        use_builtin_primary_gradient_fn=False,
        primary_objective=primary_objective,
        initial_solution_fn=None,
        parse_trees=parse_trees,
        optimization_technique="barrier_function",
        optimizer="Powell",
        optimization_hyperparams={
            "maxiter": 1000,
            "seed": rseed,
            "hyper_search": None,
            "verbose": True,
        },
    )

    # # Run seldonian algorithm
    with pytest.raises(NotImplementedError) as excinfo:
        SA = SeldonianAlgorithm(spec)
        passed_safety, solution = SA.run(debug=True)
    error_str = (
        "Optimizer: Powell "
        "is not supported for reinforcement learning. "
        "Try optimizer='CMA-ES' instead."
    )

    assert error_str in str(excinfo.value)


def test_RL_gridworld_alt_rewards(RL_gridworld_dataset_alt_rewards):
    """Test that we can put constraints on returns that use alternate rewards"""
    rseed = 99
    np.random.seed(rseed)

    # Vanilla IS first
    IS_constraint_strs = ["-0.25 - J_pi_new_IS_[1]"]
    deltas = [0.05]

    IS_parse_trees = make_parse_trees_from_constraints(
        IS_constraint_strs,
        deltas,
        regime="reinforcement_learning",
        sub_regime="all",
        columns=[],
        delta_weight_method="equal",
    )
    (
        dataset,
        policy,
        env_kwargs,
        primary_objective,
    ) = RL_gridworld_dataset_alt_rewards()

    frac_data_in_safety = 0.6
    model = RL_model(policy=policy, env_kwargs=env_kwargs)
    # Create spec object
    IS_spec = RLSpec(
        dataset=dataset,
        model=model,
        frac_data_in_safety=frac_data_in_safety,
        use_builtin_primary_gradient_fn=False,
        primary_objective=primary_objective,
        initial_solution_fn=None,
        parse_trees=IS_parse_trees,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": 0.5,
            "alpha_theta": 0.01,
            "alpha_lamb": 0.01,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 5,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # # Run seldonian algorithm
    IS_SA = SeldonianAlgorithm(IS_spec)
    passed_safety, solution = IS_SA.run()

    ## now PDIS
    PDIS_constraint_strs = ["-0.25 - J_pi_new_PDIS_[1]"]
    deltas = [0.05]

    PDIS_parse_trees = make_parse_trees_from_constraints(
        PDIS_constraint_strs,
        deltas,
        regime="reinforcement_learning",
        sub_regime="all",
        columns=[],
        delta_weight_method="equal",
    )

    PDIS_spec = RLSpec(
        dataset=dataset,
        model=model,
        frac_data_in_safety=frac_data_in_safety,
        use_builtin_primary_gradient_fn=False,
        primary_objective=primary_objective,
        initial_solution_fn=None,
        parse_trees=PDIS_parse_trees,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": 0.5,
            "alpha_theta": 0.01,
            "alpha_lamb": 0.01,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 5,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # # Run seldonian algorithm
    PDIS_SA = SeldonianAlgorithm(PDIS_spec)
    passed_safety, solution = PDIS_SA.run()


def test_RL_gridworld_addl_dataset(RL_gridworld_addl_dataset):
    """Test that the RL gridworld example runs
    with a simple performance improvement constraint. Make
    sure safety test passes and solution is correct.
    """

    # IS estimate
    # Load data and metadata
    rseed = 99
    np.random.seed(rseed)
    constraint_strs = ["-10.0 - J_pi_new_IS"]
    deltas = [0.05]

    (
        primary_dataset,
        additional_datasets,
        model,
        primary_objective,
        parse_trees,
    ) = RL_gridworld_addl_dataset(constraint_strs, deltas)

    frac_data_in_safety = 0.6

    # Create spec object
    spec = RLSpec(
        dataset=primary_dataset,
        additional_datasets=additional_datasets,
        model=model,
        frac_data_in_safety=frac_data_in_safety,
        use_builtin_primary_gradient_fn=False,
        primary_objective=primary_objective,
        parse_trees=parse_trees,
        initial_solution_fn=None,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": 0.5,
            "alpha_theta": 0.01,
            "alpha_lamb": 0.01,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 5,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # # Run seldonian algorithm
    SA = SeldonianAlgorithm(spec)
    # Ensure that the candidate and safety datasets were created within the additional_datasets object
    for pt in spec.parse_trees:
        base_nodes_this_tree = list(pt.base_node_dict.keys())
        for bn in base_nodes_this_tree:
            assert "candidate_dataset" in additional_datasets[pt.constraint_str][bn]
            assert "safety_dataset" in additional_datasets[pt.constraint_str][bn]
            bn_dataset = additional_datasets[pt.constraint_str][bn]["dataset"]
            assert bn_dataset.num_datapoints == 50
            cd = additional_datasets[pt.constraint_str][bn]["candidate_dataset"]
            assert cd.num_datapoints == int(round((1.0 - frac_data_in_safety) * 50))
            sd = additional_datasets[pt.constraint_str][bn]["safety_dataset"]
            assert sd.num_datapoints == int(round((frac_data_in_safety) * 50))

    # Run the Seldonian algorithm
    passed_safety, solution = SA.run()
    assert passed_safety == True
    array_to_compare = np.array(
        [
            [0.07390434, -0.0719768, -0.0724801, 0.07081302,],
            [0.07195114, 0.07210082, -0.07183858, 0.07117908,],
            [0.07163089, -0.07224132, -0.07171612, 0.07213358,],
            [-0.0718277, 0.07132247, -0.07168555, 0.07225085,],
            [-0.07232368, 0.07285194, -0.07313774, 0.07051624,],
            [-0.07368507, -0.07004692, -0.07180388, 0.07248678,],
            [-0.07194876, 0.07323111, 0.0717594, 0.0721216,],
            [0.07234534, 0.07180869, -0.07459196, -0.07171865,],
            [0.0, 0.0, 0.0, 0.0,],
        ]
    )

    assert np.allclose(solution, array_to_compare)
