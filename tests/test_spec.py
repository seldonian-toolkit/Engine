import os
import pytest
import importlib
import pandas as pd

from seldonian.utils.io_utils import load_json
from seldonian.dataset import DataSetLoader, SupervisedDataSet, RLDataSet
from seldonian.parse_tree.parse_tree import make_parse_trees_from_constraints
from seldonian.spec import *


def test_createSupervisedSpec(spec_garbage_collector):
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

    constraint_strs = ["Mean_Squared_Error - 2.0", "Mean_Squared_Error - 2.0"]
    deltas = [0.05, 0.05]
    dataset, model, primary_objective, parse_trees = gpa_regression_dataset(
            constraint_strs,deltas) 
    with pytest.raises(RuntimeError) as excinfo:
        spec = SupervisedSpec(
            dataset=dataset,
            model=model,
            primary_objective=primary_objective,
            parse_trees=parse_trees,
            sub_regime="regression",
        )
    error_str = (
        "The constraint: 'Mean_Squared_Error - 2.0' "
        "appears more than once in the list of constraints. "
        "Duplicate constraints are not allowed."
    )
    assert str(excinfo.value) == error_str


def test_supervised_spec_additional_datasets(gpa_regression_dataset):
    # Test 1: create a spec object using a single primary dataset and a single additional dataset
    # This is the simplest possible case where we have no custom candidate/safety datasets
    # and no missing parse trees or base nodes from the additional datasets dict
    base_node = "Mean_Squared_Error"
    constraint_str = f"{base_node} - 2.0"
    constraint_strs = [constraint_str]
    deltas = [0.5]
    dataset, model, primary_objective, parse_trees = gpa_regression_dataset(
        constraint_strs,deltas) 

    num_datapoints=1000
    new_features = dataset.features[0:num_datapoints]
    new_labels = dataset.labels[0:num_datapoints]
    new_sensitive_attrs = dataset.sensitive_attrs[0:num_datapoints]
    new_dataset = SupervisedDataSet(
        features=new_features,
        labels=new_labels,
        sensitive_attrs=new_sensitive_attrs,
        num_datapoints=num_datapoints,
        meta=dataset.meta
    )
    additional_datasets = {}
    additional_datasets[constraint_str] = {
        base_node: {
            "dataset": new_dataset
        }
    }

    spec = SupervisedSpec(
            dataset=dataset,
            model=model,
            primary_objective=primary_objective,
            parse_trees=parse_trees,
            sub_regime="regression",
            additional_datasets=additional_datasets
    )

    assert constraint_str in spec.additional_datasets
    assert "dataset" in spec.additional_datasets[constraint_str][base_node] 
    extracted_dataset = spec.additional_datasets[constraint_str][base_node]["dataset"]
    assert extracted_dataset.features.shape == (1000,9)
    assert extracted_dataset.labels.shape == (1000,)
    assert extracted_dataset.num_datapoints == num_datapoints

    # Test 2: An error is raised if we use a bad constraint string as a key
    bad_constraint_str = "Bad_Base_Node - 10.0"
    additional_datasets = {}
    additional_datasets[bad_constraint_str] = {
        base_node: {
            "dataset": new_dataset
        }
    }
    with pytest.raises(RuntimeError) as excinfo:
        spec = SupervisedSpec(
                dataset=dataset,
                model=model,
                primary_objective=primary_objective,
                parse_trees=parse_trees,
                sub_regime="regression",
                additional_datasets=additional_datasets
        )
    error_str = (
        f"The constraint: '{bad_constraint_str}' "
        "does not match the constraint strings found in the parse trees:"
        f"{{'Mean_Squared_Error - 2.0'}}. "
        "Check formatting."
    )
    assert str(excinfo.value) == error_str
    
    # Test 3: An error is raised if we use a bad base node as a key 

    additional_datasets = {}
    additional_datasets[constraint_str] = {
        "Bad_Base_Node": {
            "dataset": new_dataset
        }
    }
    with pytest.raises(RuntimeError) as excinfo:
        spec = SupervisedSpec(
                dataset=dataset,
                model=model,
                primary_objective=primary_objective,
                parse_trees=parse_trees,
                sub_regime="regression",
                additional_datasets=additional_datasets
        )
    error_str = (
         f"The base node: 'Bad_Base_Node' "
        "does not match the base nodes found in this parse tree:"
        "['Mean_Squared_Error']. "
        "Check formatting."
    )
    assert str(excinfo.value) == error_str

    # Test 4: An error is raised if the "dataset" key is present 
    # and the "candidate_dataset" or "safety_dataset" key is also present

    additional_datasets = {}
    additional_datasets[constraint_str] = {
        base_node: {
            "dataset": new_dataset,
            "candidate_dataset": new_dataset,
            "safety_dataset": new_dataset
        }
    }
    with pytest.raises(RuntimeError) as excinfo:
        spec = SupervisedSpec(
                dataset=dataset,
                model=model,
                primary_objective=primary_objective,
                parse_trees=parse_trees,
                sub_regime="regression",
                additional_datasets=additional_datasets
        )
    error_str = (
        f"There is an issue with the additional_datasets['{constraint_str}']['{base_node}'] dictionary. "
        "'dataset' key is present, so 'candidate_dataset' and 'safety_dataset' keys cannot be present. "
    )
    assert str(excinfo.value) == error_str

    # Test 5: An error is raised if the "dataset" key is missing, 
    # and the "candidate_dataset" and "safety_dataset" keys are not both present
    additional_datasets = {}
    additional_datasets[constraint_str] = {
        base_node: {
            "candidate_dataset": new_dataset,
        }
    }
    with pytest.raises(RuntimeError) as excinfo:
        spec = SupervisedSpec(
                dataset=dataset,
                model=model,
                primary_objective=primary_objective,
                parse_trees=parse_trees,
                sub_regime="regression",
                additional_datasets=additional_datasets
        )
    error_str = (
       f"There is an issue with the additional_datasets['{constraint_str}']['{base_node}'] dictionary. "
        "'dataset' key is not present, so 'candidate_dataset' and 'safety_dataset' keys must be present. "
    )
    assert str(excinfo.value) == error_str

    # NEW SET OF TESTS: One parse tree, two base nodes

    base_node1 = "Mean_Squared_Error"
    base_node2 = "Mean_Error"
    constraint_str = f"{base_node1} + {base_node2}  - 5.0"
    constraint_strs = [constraint_str]
    deltas = [0.5]
    _, _, _, parse_trees = gpa_regression_dataset(
        constraint_strs,deltas) 
    
    # Add a second dataset
    num_datapoints2=2000
    new_features2 = dataset.features[0:num_datapoints2]
    new_labels2 = dataset.labels[0:num_datapoints2]
    new_sensitive_attrs2 = dataset.sensitive_attrs[0:num_datapoints2]
    new_dataset2 = SupervisedDataSet(
        features=new_features2,
        labels=new_labels2,
        sensitive_attrs=new_sensitive_attrs2,
        num_datapoints=num_datapoints2,
        meta=dataset.meta
    )
    # Test 6: Including both base nodes works
    additional_datasets = {}
    additional_datasets[constraint_str] = {
        base_node1: {
            "dataset": new_dataset
        },
        base_node2: {
            "dataset": new_dataset2
        }
    }

    spec = SupervisedSpec(
            dataset=dataset,
            model=model,
            primary_objective=primary_objective,
            parse_trees=parse_trees,
            sub_regime="regression",
            additional_datasets=additional_datasets
    )

    assert constraint_str in spec.additional_datasets
    assert "dataset" in spec.additional_datasets[constraint_str][base_node1] 
    extracted_dataset1 = spec.additional_datasets[constraint_str][base_node1]["dataset"]
    assert extracted_dataset1.features.shape == (1000,9)
    assert extracted_dataset1.labels.shape == (1000,)
    assert extracted_dataset1.num_datapoints == num_datapoints

    assert "dataset" in spec.additional_datasets[constraint_str][base_node2] 
    extracted_dataset2 = spec.additional_datasets[constraint_str][base_node2]["dataset"]
    assert extracted_dataset2.features.shape == (2000,9)
    assert extracted_dataset2.labels.shape == (2000,)
    assert extracted_dataset2.num_datapoints == num_datapoints2


    # Test 6: Leaving out one base nodes results in that base node getting assigned the primary dataset
    additional_datasets = {}
    additional_datasets[constraint_str] = {
        base_node1: {
            "dataset": new_dataset
        },
    }

    spec = SupervisedSpec(
            dataset=dataset,
            model=model,
            primary_objective=primary_objective,
            parse_trees=parse_trees,
            sub_regime="regression",
            additional_datasets=additional_datasets
    )

    assert constraint_str in spec.additional_datasets
    assert "dataset" in spec.additional_datasets[constraint_str][base_node1] 
    extracted_dataset1 = spec.additional_datasets[constraint_str][base_node1]["dataset"]
    assert extracted_dataset1.features.shape == (1000,9)
    assert extracted_dataset1.labels.shape == (1000,)
    assert extracted_dataset1.num_datapoints == num_datapoints

    assert "dataset" in spec.additional_datasets[constraint_str][base_node2] 
    extracted_dataset2 = spec.additional_datasets[constraint_str][base_node2]["dataset"]
    assert extracted_dataset2.features.shape == (43303,9)
    assert extracted_dataset2.labels.shape == (43303,)
    assert extracted_dataset2.num_datapoints == 43303

    # Test 7: It is valid for one base node to get a single dataset 
    # and the second to get candidate/safety datasets
    additional_datasets = {}
    additional_datasets[constraint_str] = {
        base_node1: {
            "dataset": new_dataset
        },
        base_node2: {
            "candidate_dataset": new_dataset,
            "safety_dataset": new_dataset2
        },
    }

    spec = SupervisedSpec(
            dataset=dataset,
            model=model,
            primary_objective=primary_objective,
            parse_trees=parse_trees,
            sub_regime="regression",
            additional_datasets=additional_datasets
    )

    assert constraint_str in spec.additional_datasets
    assert "dataset" in spec.additional_datasets[constraint_str][base_node1] 
    extracted_dataset1 = spec.additional_datasets[constraint_str][base_node1]["dataset"]
    assert extracted_dataset1.features.shape == (1000,9)
    assert extracted_dataset1.labels.shape == (1000,)
    assert extracted_dataset1.num_datapoints == num_datapoints

    assert "dataset" not in spec.additional_datasets[constraint_str][base_node2] 
    assert "candidate_dataset" in spec.additional_datasets[constraint_str][base_node2] 
    assert "safety_dataset" in spec.additional_datasets[constraint_str][base_node2] 
    extracted_candidate_dataset2 = spec.additional_datasets[constraint_str][base_node2]["candidate_dataset"]
    assert extracted_candidate_dataset2.features.shape == (1000,9)
    assert extracted_candidate_dataset2.labels.shape == (1000,)
    assert extracted_candidate_dataset2.num_datapoints == 1000
    extracted_safety_dataset2 = spec.additional_datasets[constraint_str][base_node2]["safety_dataset"]
    assert extracted_safety_dataset2.features.shape == (2000,9)
    assert extracted_safety_dataset2.labels.shape == (2000,)
    assert extracted_safety_dataset2.num_datapoints == 2000


    # NEW SET OF TESTS: Two parse trees, two base nodes on each

    base_node1 = "Mean_Squared_Error"
    base_node2 = "Mean_Error"
    constraint_str1 = f"{base_node1} + {base_node2}  - 15.0"
    constraint_str2 = f"max({base_node1},{base_node2})  - 2.0"
    constraint_strs = [constraint_str1,constraint_str2]
    deltas = [0.05,0.2]
    _, _, _, parse_trees = gpa_regression_dataset(
        constraint_strs,deltas) 

    # Test 8: If we only provide a single base node for a single tree,
    # then other base node of that tree gets primary dataset
    # and both base nodes in other tree get primary dataset

    additional_datasets = {}
    additional_datasets[constraint_str1] = {
        base_node1: {
            "dataset": new_dataset
        },
    }

    spec = SupervisedSpec(
            dataset=dataset,
            model=model,
            primary_objective=primary_objective,
            parse_trees=parse_trees,
            sub_regime="regression",
            additional_datasets=additional_datasets
    )


    assert constraint_str1 in spec.additional_datasets
    assert constraint_str2 in spec.additional_datasets

    assert "dataset" in spec.additional_datasets[constraint_str1][base_node1] 
    extracted_dataset1 = spec.additional_datasets[constraint_str1][base_node1]["dataset"]
    assert extracted_dataset1.features.shape == (1000,9)
    assert extracted_dataset1.labels.shape == (1000,)
    assert extracted_dataset1.num_datapoints == num_datapoints

    assert "dataset" in spec.additional_datasets[constraint_str1][base_node2] 
    extracted_dataset2 = spec.additional_datasets[constraint_str1][base_node2]["dataset"]
    assert extracted_dataset2.features.shape == (43303,9)
    assert extracted_dataset2.labels.shape == (43303,)
    assert extracted_dataset2.num_datapoints == 43303

    assert "dataset" in spec.additional_datasets[constraint_str2][base_node1] 
    extracted_dataset1 = spec.additional_datasets[constraint_str2][base_node1]["dataset"]
    assert extracted_dataset1.features.shape == (43303,9)
    assert extracted_dataset1.labels.shape == (43303,)
    assert extracted_dataset1.num_datapoints == 43303

    assert "dataset" in spec.additional_datasets[constraint_str2][base_node2] 
    extracted_dataset2 = spec.additional_datasets[constraint_str2][base_node2]["dataset"]
    assert extracted_dataset2.features.shape == (43303,9)
    assert extracted_dataset2.labels.shape == (43303,)
    assert extracted_dataset2.num_datapoints == 43303



    # Test 9: Provide custom datasets for each base node in each tree

    additional_datasets = {}
    additional_datasets[constraint_str1] = {
        base_node1: {
            "dataset": new_dataset2
        },
        base_node2: {
            "dataset": new_dataset
        },
    }

    additional_datasets[constraint_str2] = {
        base_node1: {
            "dataset": dataset
        },
        base_node2: {
            "dataset": new_dataset2
        },
    }

    spec = SupervisedSpec(
            dataset=dataset,
            model=model,
            primary_objective=primary_objective,
            parse_trees=parse_trees,
            sub_regime="regression",
            additional_datasets=additional_datasets
    )


    assert constraint_str1 in spec.additional_datasets
    assert constraint_str2 in spec.additional_datasets

    assert "dataset" in spec.additional_datasets[constraint_str1][base_node1] 
    extracted_dataset1 = spec.additional_datasets[constraint_str1][base_node1]["dataset"]
    assert extracted_dataset1.features.shape == (2000,9)
    assert extracted_dataset1.labels.shape == (2000,)
    assert extracted_dataset1.num_datapoints == 2000

    assert "dataset" in spec.additional_datasets[constraint_str1][base_node2] 
    extracted_dataset2 = spec.additional_datasets[constraint_str1][base_node2]["dataset"]
    assert extracted_dataset2.features.shape == (1000,9)
    assert extracted_dataset2.labels.shape == (1000,)
    assert extracted_dataset2.num_datapoints == 1000

    assert "dataset" in spec.additional_datasets[constraint_str2][base_node1] 
    extracted_dataset1 = spec.additional_datasets[constraint_str2][base_node1]["dataset"]
    assert extracted_dataset1.features.shape == (43303,9)
    assert extracted_dataset1.labels.shape == (43303,)
    assert extracted_dataset1.num_datapoints == 43303

    assert "dataset" in spec.additional_datasets[constraint_str2][base_node2] 
    extracted_dataset2 = spec.additional_datasets[constraint_str2][base_node2]["dataset"]
    assert extracted_dataset2.features.shape == (2000,9)
    assert extracted_dataset2.labels.shape == (2000,)
    assert extracted_dataset2.num_datapoints == 2000

    # NEW SET OF TESTS: Custom candidate/safety for primary dataset 

    candidate_dataset = new_dataset
    safety_dataset = new_dataset2

    # Test 10: One tree with one base node. Its additional dataset overrides the primary candidate/safety split
    base_node = "Mean_Squared_Error"
    constraint_str = f"{base_node} - 2.0"
    constraint_strs = [constraint_str]
    deltas = [0.05]
    _, _, _, parse_trees = gpa_regression_dataset(
        constraint_strs,deltas) 

    additional_datasets = {}
    additional_datasets[constraint_str] = {
        base_node: {
            "dataset": dataset
        }
    }
        
    spec = SupervisedSpec(
            dataset=None,
            candidate_dataset=candidate_dataset,
            safety_dataset=safety_dataset,
            model=model,
            primary_objective=primary_objective,
            parse_trees=parse_trees,
            sub_regime="regression",
            additional_datasets=additional_datasets
    )

    assert spec.candidate_dataset.features.shape == (1000,9)
    assert spec.safety_dataset.features.shape == (2000,9)
    assert constraint_str in spec.additional_datasets

    assert "dataset" in spec.additional_datasets[constraint_str][base_node] 
    assert "candidate_dataset" not in spec.additional_datasets[constraint_str][base_node] 
    assert "safety_dataset" not in spec.additional_datasets[constraint_str][base_node] 
    extracted_dataset1 = spec.additional_datasets[constraint_str][base_node]["dataset"]
    assert extracted_dataset1.features.shape == (43303,9)
    assert extracted_dataset1.labels.shape == (43303,)
    assert extracted_dataset1.num_datapoints == 43303

    # Test 10: Two trees with two base nodes. Provide additional dataset for one of them in one tree only. The other gets the primary candidate/safety datasets.

    base_node1 = "Mean_Squared_Error"
    base_node2 = "Mean_Error"
    constraint_str1 = f"{base_node1} + {base_node2}  - 15.0"
    constraint_str2 = f"max({base_node1},{base_node2})  - 2.0"
    constraint_strs = [constraint_str1,constraint_str2]
    deltas = [0.05,0.2]
    _, _, _, parse_trees = gpa_regression_dataset(
        constraint_strs,deltas) 

    additional_datasets = {}
    additional_datasets[constraint_str1] = {
        base_node1: {
            "dataset": dataset
        },

    }
        
    spec = SupervisedSpec(
            dataset=None,
            candidate_dataset=candidate_dataset,
            safety_dataset=safety_dataset,
            model=model,
            primary_objective=primary_objective,
            parse_trees=parse_trees,
            sub_regime="regression",
            additional_datasets=additional_datasets
    )

    assert spec.candidate_dataset.features.shape == (1000,9)
    assert spec.safety_dataset.features.shape == (2000,9)
    assert constraint_str1 in spec.additional_datasets
    assert constraint_str2 in spec.additional_datasets

    assert "dataset" in spec.additional_datasets[constraint_str1][base_node1] 
    assert "candidate_dataset" not in spec.additional_datasets[constraint_str1][base_node1] 
    assert "safety_dataset" not in spec.additional_datasets[constraint_str1][base_node1] 
    extracted_dataset1 = spec.additional_datasets[constraint_str1][base_node1]["dataset"]
    assert extracted_dataset1.features.shape == (43303,9)
    assert extracted_dataset1.labels.shape == (43303,)
    assert extracted_dataset1.num_datapoints == 43303

    assert "dataset" not in spec.additional_datasets[constraint_str1][base_node2] 
    assert "candidate_dataset" in spec.additional_datasets[constraint_str1][base_node2] 
    assert "safety_dataset" in spec.additional_datasets[constraint_str1][base_node2] 
    extracted_candidate_dataset1 = spec.additional_datasets[constraint_str1][base_node2]["candidate_dataset"]
    assert extracted_candidate_dataset1.features.shape == (1000,9)
    assert extracted_candidate_dataset1.labels.shape == (1000,)
    assert extracted_candidate_dataset1.num_datapoints == 1000
    extracted_safety_dataset1 = spec.additional_datasets[constraint_str1][base_node2]["safety_dataset"]
    assert extracted_safety_dataset1.features.shape == (2000,9)
    assert extracted_safety_dataset1.labels.shape == (2000,)
    assert extracted_safety_dataset1.num_datapoints == 2000

    for bn in [base_node1,base_node2]:
        assert "dataset" not in spec.additional_datasets[constraint_str2][bn] 
        assert "candidate_dataset" in spec.additional_datasets[constraint_str2][bn] 
        assert "safety_dataset" in spec.additional_datasets[constraint_str2][bn] 
        extracted_candidate_dataset2 = spec.additional_datasets[constraint_str2][bn]["candidate_dataset"]
        assert extracted_candidate_dataset2.features.shape == (1000,9)
        assert extracted_candidate_dataset2.labels.shape == (1000,)
        assert extracted_candidate_dataset2.num_datapoints == 1000
        extracted_safety_dataset2 = spec.additional_datasets[constraint_str2][bn]["safety_dataset"]
        assert extracted_safety_dataset2.features.shape == (2000,9)
        assert extracted_safety_dataset2.labels.shape == (2000,)
        assert extracted_safety_dataset2.num_datapoints == 2000

    


