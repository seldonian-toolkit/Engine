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

from seldonian.spec import *
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.models.models import *
from seldonian.models import objectives
from seldonian.RL.RL_model import RL_model

### Begin tests


def test_addl_dataset_batches(gpa_classification_addl_datasets):
    """
    """
    # Load metadata
    rseed = 0
    np.random.seed(rseed)
    constraint_strs = ["abs((PR | [M]) - (PR | [F])) - 0.15"]
    deltas = [0.05]

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
            "use_batches": True,
            "n_epochs": 5,
            "batch_size": primary_batch_size,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Set up SA object
    SA = SeldonianAlgorithm(spec)
    SA.set_initial_solution()

    CS = SA.candidate_selection()

    # Calculate the start/end indicies of each batch in the additional datasets
    CS.precalculate_addl_dataset_batch_indices(
        n_epochs=1, n_batches=4, primary_batch_size=primary_batch_size
    )

    for cstr in CS.additional_datasets:
        for bn in CS.additional_datasets[cstr]:
            this_dict = CS.additional_datasets[cstr][bn]
            if bn == "PR | [M]":
                assert this_dict["batch_index_list"] == [
                    [0, 1000],
                    [1000, 2000],
                    [2000, 3000],
                    [3000, 4000],
                ]
            elif bn == "PR | [F]":
                assert this_dict["batch_index_list"] == [
                    [0, 5000],
                    [5000, 10000],
                    [10000, 13857, 0, 1143],
                    [1143, 6143],
                ]

    # Do it again with multiple epochs
    CS.precalculate_addl_dataset_batch_indices(
        n_epochs=3, n_batches=3, primary_batch_size=primary_batch_size
    )
    batch_index_dict = {
        "PR | [M]": [
            [0, 1000],
            [1000, 2000],
            [2000, 3000],
            [3000, 4000],
            [4000, 5000],
            [5000, 6000],
            [6000, 7000],
            [7000, 8000],
            [8000, 9000],
        ],
        "PR | [F]": [
            [0, 5000],
            [5000, 10000],
            [10000, 13857, 0, 1143],
            [1143, 6143],
            [6143, 11143],
            [11143, 13857, 0, 2286],
            [2286, 7286],
            [7286, 12286],
            [12286, 13857, 0, 3429],
        ],
    }
    for cstr in CS.additional_datasets:
        for bn in CS.additional_datasets[cstr]:
            this_dict = CS.additional_datasets[cstr][bn]
            assert this_dict["batch_index_list"] == batch_index_dict[bn]

    # Now calculate the batches using these indices and verify that they are the correct size
    CS.calculate_batches_addl_datasets(
        primary_epoch_index=0, primary_batch_index=0, n_batches=3
    )
    for cstr in CS.additional_datasets:
        for bn in CS.additional_datasets[cstr]:
            this_dict = CS.additional_datasets[cstr][bn]
            batch_dataset = this_dict["batch_dataset"]
            batch_index_list = batch_index_dict[bn]
            batch_indices = batch_index_list[0]

            assert len(batch_dataset.features) == batch_indices[-1]
            assert len(batch_dataset.labels) == batch_indices[-1]
            assert len(batch_dataset.sensitive_attrs) == batch_indices[-1]
            assert batch_dataset.num_datapoints == batch_indices[-1]

    # Now calculate the batches using these indices and verify that they are the correct size
    CS.calculate_batches_addl_datasets(
        primary_epoch_index=0, primary_batch_index=2, n_batches=3
    )
    for cstr in CS.additional_datasets:
        for bn in CS.additional_datasets[cstr]:
            this_dict = CS.additional_datasets[cstr][bn]
            batch_dataset = this_dict["batch_dataset"]
            batch_index_list = batch_index_dict[bn]
            batch_indices = batch_index_list[0]
            if len(batch_indices) == 4:
                assert len(batch_dataset.features) == (
                    batch_indices[1]
                    - batch_indices[0]
                    + batch_indices[3]
                    - batch_indices[2]
                )
                assert len(batch_dataset.labels) == (
                    batch_indices[1]
                    - batch_indices[0]
                    + batch_indices[3]
                    - batch_indices[2]
                )
                assert len(batch_dataset.sensitive_attrs) == (
                    batch_indices[1]
                    - batch_indices[0]
                    + batch_indices[3]
                    - batch_indices[2]
                )
                assert batch_dataset.num_datapoints == (
                    batch_indices[1]
                    - batch_indices[0]
                    + batch_indices[3]
                    - batch_indices[2]
                )
            else:
                assert len(batch_dataset.features) == (
                    batch_indices[1] - batch_indices[0]
                )
                assert len(batch_dataset.labels) == (
                    batch_indices[1] - batch_indices[0]
                )
                assert len(batch_dataset.sensitive_attrs) == (
                    batch_indices[1] - batch_indices[0]
                )
                assert batch_dataset.num_datapoints == (
                    batch_indices[1] - batch_indices[0]
                )

    # Test that an error is raised if we try to initiate with a batch size that will be larger than n_candidate

    primary_batch_size = 5000
    batch_size_dict = {"abs((PR | [M]) - (PR | [F])) - 0.15": {"PR | [M]": 20000,}}

    (
        primary_dataset,
        additional_datasets,
        model,
        primary_objective,
        parse_trees,
    ) = gpa_classification_addl_datasets(
        constraint_strs=constraint_strs, deltas=deltas, batch_size_dict=batch_size_dict
    )

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
            "use_batches": True,
            "n_epochs": 5,
            "batch_size": primary_batch_size,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": True,
        },
    )

    # Set up SA object
    with pytest.raises(RuntimeError) as excinfo:
        SA = SeldonianAlgorithm(spec)

    error_str = (
        "additional_datasets['abs((PR | [M]) - (PR | [F])) - 0.15']['PR | [M]']['batch_size'] = 20000, "
        "which is larger than the number of data points in the candidate dataset: 13857 after splitting."
    )
    assert str(excinfo.value) == error_str
