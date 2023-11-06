import os
import pytest
import importlib
import pandas as pd

from seldonian.utils.io_utils import load_json
from seldonian.dataset import DataSetLoader, SupervisedDataSet, RLDataSet
from seldonian.parse_tree.parse_tree import make_parse_trees_from_constraints
from seldonian.spec import createSupervisedSpec, createRLSpec


def test_createSupervisedSpec(gpa_regression_dataset, spec_garbage_collector):
    """Test that if the user does not provide a primary objective,
    then the default is used in the three different regimes/sub-regimes
    """
    # Regression
    data_pth = "static/datasets/supervised/GPA/gpa_regression_dataset.csv"
    metadata_pth = "static/datasets/supervised/GPA/metadata_regression.json"

    metadata_dict = load_json(metadata_pth)
    regime = metadata_dict["regime"]
    sub_regime = metadata_dict["sub_regime"]
    all_col_names = metadata_dict["all_col_names"]
    sensitive_col_names = metadata_dict["sensitive_col_names"]

    include_sensitive_columns = False
    regime = "supervised_learning"

    # Load dataset from file
    loader = DataSetLoader(regime=regime)

    dataset = loader.load_supervised_dataset(
        filename=data_pth, metadata_filename=metadata_pth, file_type="csv"
    )

    constraint_strs = ["Mean_Squared_Error - 2.0"]
    deltas = [0.05]
    save_dir = "tests/specfiles"
    spec_savename = os.path.join(save_dir, "spec.pkl")
    assert not os.path.exists(spec_savename)
    spec = createSupervisedSpec(
        dataset=dataset,
        metadata_pth=metadata_pth,
        constraint_strs=constraint_strs,
        deltas=deltas,
        save=True,
        save_dir=save_dir,
    )
    assert os.path.exists(spec_savename)
    os.remove(spec_savename)
    assert not os.path.exists(spec_savename)
    assert spec.primary_objective != None
    assert spec.primary_objective.__name__ == "Mean_Squared_Error"
    assert len(spec.parse_trees) == 1

    # Binary Classification
    data_pth = "static/datasets/supervised/GPA/gpa_classification_dataset.csv"
    metadata_pth = "static/datasets/supervised/GPA/metadata_classification.json"

    metadata_dict = load_json(metadata_pth)
    regime = metadata_dict["regime"]
    sub_regime = metadata_dict["sub_regime"]
    all_col_names = metadata_dict["all_col_names"]
    sensitive_col_names = metadata_dict["sensitive_col_names"]

    regime = "supervised_learning"

    # Load dataset from file
    loader = DataSetLoader(regime=regime)

    dataset = loader.load_supervised_dataset(
        filename=data_pth, metadata_filename=metadata_pth, file_type="csv"
    )

    constraint_strs = ["FPR - 0.5"]
    deltas = [0.05]

    spec_savename = os.path.join(save_dir, "spec.pkl")
    assert not os.path.exists(spec_savename)

    spec = createSupervisedSpec(
        dataset=dataset,
        metadata_pth=metadata_pth,
        constraint_strs=constraint_strs,
        deltas=deltas,
        save=True,
        save_dir=save_dir,
    )
    assert os.path.exists(spec_savename)
    os.remove(spec_savename)
    assert not os.path.exists(spec_savename)

    assert spec.primary_objective != None
    assert spec.primary_objective.__name__ == "binary_logistic_loss"
    assert len(spec.parse_trees) == 1

    # Multiclass Classification
    data_pth = "static/datasets/supervised/GPA/gpa_multiclass_dataset.csv"
    metadata_pth = "static/datasets/supervised/GPA/metadata_multiclass.json"

    metadata_dict = load_json(metadata_pth)
    regime = metadata_dict["regime"]
    sub_regime = metadata_dict["sub_regime"]
    all_col_names = metadata_dict["all_col_names"]
    sensitive_col_names = metadata_dict["sensitive_col_names"]

    regime = "supervised_learning"

    # Load dataset from file
    loader = DataSetLoader(regime=regime)

    dataset = loader.load_supervised_dataset(
        filename=data_pth, metadata_filename=metadata_pth, file_type="csv"
    )

    constraint_strs = ["CM_[0,0] - 0.5"]
    deltas = [0.05]

    spec_savename = os.path.join(save_dir, "spec.pkl")
    assert not os.path.exists(spec_savename)

    spec = createSupervisedSpec(
        dataset=dataset,
        metadata_pth=metadata_pth,
        constraint_strs=constraint_strs,
        deltas=deltas,
        save=True,
        save_dir=save_dir,
    )
    assert os.path.exists(spec_savename)
    os.remove(spec_savename)
    assert not os.path.exists(spec_savename)

    assert spec.primary_objective != None
    assert spec.primary_objective.__name__ == "multiclass_logistic_loss"
    assert len(spec.parse_trees) == 1


def test_createRLSpec(RL_gridworld_dataset, spec_garbage_collector):
    """Test that if the user does not provide a primary objective,
    then the default is used in the three different regimes/sub-regimes
    """
    # Regression

    constraint_strs = ["J_pi_new_IS >= -0.25"]
    deltas = [0.05]

    save_dir = "tests/specfiles"
    spec_savename = os.path.join(save_dir, "spec.pkl")
    assert not os.path.exists(spec_savename)
    (dataset, policy, env_kwargs, primary_objective) = RL_gridworld_dataset()

    spec = createRLSpec(
        dataset=dataset,
        policy=policy,
        env_kwargs={},
        constraint_strs=constraint_strs,
        deltas=deltas,
        frac_data_in_safety=0.6,
        initial_solution_fn=None,
        use_builtin_primary_gradient_fn=False,
        save=True,
        save_dir=save_dir,
    )
    assert os.path.exists(spec_savename)
    assert spec.primary_objective != None
    assert spec.primary_objective.__name__ == "IS_estimate"
    assert len(spec.parse_trees) == 1


def test_duplicate_parse_trees(gpa_regression_dataset):
    """Test that entering the same constraint more than once
    raises an error when making the spec object
    """
    # Regression
    data_pth = "static/datasets/supervised/GPA/gpa_regression_dataset.csv"
    metadata_pth = "static/datasets/supervised/GPA/metadata_regression.json"

    metadata_dict = load_json(metadata_pth)
    regime = metadata_dict["regime"]
    sub_regime = metadata_dict["sub_regime"]
    all_col_names = metadata_dict["all_col_names"]
    sensitive_col_names = metadata_dict["sensitive_col_names"]

    regime = "supervised_learning"

    # Load dataset from file
    loader = DataSetLoader(regime=regime)

    dataset = loader.load_supervised_dataset(
        filename=data_pth, metadata_filename=metadata_pth, file_type="csv"
    )

    constraint_strs = ["Mean_Squared_Error - 2.0", "Mean_Squared_Error - 2.0"]
    deltas = [0.05, 0.05]

    with pytest.raises(RuntimeError) as excinfo:
        spec = createSupervisedSpec(
            dataset=dataset,
            metadata_pth=metadata_pth,
            constraint_strs=constraint_strs,
            deltas=deltas,
            save=False,
        )
    error_str = (
        "The constraint: 'Mean_Squared_Error - 2.0' "
        "appears more than once in the list of constraints. "
        "Duplicate constraints are not allowed."
    )
    assert str(excinfo.value) == error_str
