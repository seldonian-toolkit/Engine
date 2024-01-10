import os
import shutil
import autograd.numpy as np  # Thinly-wrapped version of Numpy
import pytest

from seldonian.parse_tree.parse_tree import *
from seldonian.utils.io_utils import load_json, load_pickle
from seldonian.utils.tutorial_utils import generate_data
from seldonian.dataset import *
from seldonian.spec import *
from seldonian.models import objectives
from seldonian.models.models import *


@pytest.fixture
def stump():
    def stump_function(
        operator_type,
        left_bounds,
        right_bounds,
        regime="supervised_learning",
        sub_regime="classification",
    ):
        # A parse tree with a root node and left and right children only
        root = InternalNode(operator_type)
        root.index = 0
        root.left = BaseNode("a")
        root.left.index = 1
        root.right = BaseNode("b")
        root.right.index = 2

        pt = ParseTree(delta=0.05, regime=regime, sub_regime=sub_regime)
        # pt.create_from_ast("a+b")
        pt.root = root
        pt.root.left.lower = left_bounds[0]
        pt.root.left.upper = left_bounds[1]
        pt.root.right.lower = right_bounds[0]
        pt.root.right.upper = right_bounds[1]
        pt.n_nodes = 3
        pt.n_base_nodes = 2

        pt.base_node_dict = {
            "a": {
                "bound_method": "manual",
                "bound_computed": False,
                "value_computed": False,
                "lower": float("-inf"),
                "upper": float("inf"),
                "lower_needed": None,
                "upper_needed": None,
                "delta_lower": None,
                "delta_upper": None,
                "data_dict": None,
            },
            "b": {
                "bound_method": "manual",
                "bound_computed": False,
                "value_computed": False,
                "lower": float("-inf"),
                "upper": float("inf"),
                "lower_needed": None,
                "upper_needed": None,
                "delta_lower": None,
                "delta_upper": None,
                "data_dict": None,
            },
        }
        return pt

    return stump_function


@pytest.fixture
def edge():
    def edge_function(
        operator_type,
        left_bounds,
        regime="supervised_learning",
        sub_regime="classification",
    ):
        # A parse tree with a single edge
        assert operator_type in ["abs", "exp", "log"]
        root = InternalNode(operator_type)
        root.left = BaseNode("a")
        pt = ParseTree(delta=0.05, regime=regime, sub_regime=sub_regime)
        pt.root = root
        pt.root.left.lower = left_bounds[0]
        pt.root.left.upper = left_bounds[1]
        pt.n_nodes = 2
        pt.n_base_nodes = 1
        pt.base_node_dict = {
            "a": {
                "bound_method": "manual",
                "bound_computed": False,
                "value_computed": False,
                "lower": float("-inf"),
                "upper": float("inf"),
                "lower_needed": None,
                "upper_needed": None,
                "delta_lower": None,
                "delta_upper": None,
                "data_dict": None,
            },
        }
        return pt

    return edge_function


@pytest.fixture
def spec_garbage_collector():
    save_dir = "./tests/specfiles"
    """ Fixture to create and then remove results_dir and any files it may contain"""
    print("----------- Setup spec_garbage_collector -----------")
    os.makedirs(save_dir, exist_ok=True)
    yield
    print("----------- Teardown spec_garbage_collector -----------")
    shutil.rmtree(save_dir)


@pytest.fixture
def simulated_regression_dataset_aslists():
    """Generate features, labels and sensitive attributes as lists"""
    from seldonian.models.models import LinearRegressionModel

    def generate_dataset(constraint_strs, deltas, numPoints=1000):
        regime = "supervised_learning"
        sub_regime = "regression"
        np.random.seed(0)

        model = (
            LinearRegressionModelListFeatures()
        )  # we don't have a model that supports lists of
        # features/arrays yet, but one could create one.
        X1, Y = generate_data(numPoints, loc_X=0.0, loc_Y=0.0, sigma_X=1.0, sigma_Y=1.0)
        X2 = X1**2
        meta = SupervisedMetaData(
            sub_regime="regression",
            all_col_names=["feature1", "feature2", "label"],
            feature_col_names=["feature1", "feature2"],
            label_col_names=["label"],
            sensitive_col_names=[],
        )

        # 3. Make a dataset object
        features = [np.expand_dims(X1, axis=1), np.expand_dims(X2, axis=1)]
        labels = Y
        sensitive_attrs = list()

        # Mean squared error
        primary_objective = objectives.Mean_Squared_Error

        # Load dataset from file
        loader = DataSetLoader(regime=regime)

        dataset = SupervisedDataSet(
            features=features,
            labels=labels,
            sensitive_attrs=sensitive_attrs,
            num_datapoints=numPoints,
            meta=meta,
        )

        # For each constraint, make a parse tree
        parse_trees = []
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]

            delta = deltas[ii]
            # Create parse tree object
            parse_tree = ParseTree(
                delta=delta, regime=regime, sub_regime=sub_regime, columns=[]
            )

            parse_tree.build_tree(constraint_str=constraint_str)

            parse_trees.append(parse_tree)

        return dataset, model, primary_objective, parse_trees

    return generate_dataset


@pytest.fixture
def simulated_regression_dataset():
    from seldonian.models.models import LinearRegressionModel

    def generate_dataset(constraint_strs, deltas, numPoints=1000):
        regime = "supervised_learning"
        sub_regime = "regression"
        np.random.seed(0)

        model = LinearRegressionModel()
        X, Y = generate_data(numPoints, loc_X=0.0, loc_Y=0.0, sigma_X=1.0, sigma_Y=1.0)

        meta = SupervisedMetaData(
            sub_regime="regression",
            all_col_names=["feature1", "label"],
            feature_col_names=["feature1"],
            label_col_names=["label"],
            sensitive_col_names=[],
        )

        # 3. Make a dataset object
        features = np.expand_dims(X, axis=1)
        labels = Y

        # Mean squared error
        primary_objective = objectives.Mean_Squared_Error

        # Load dataset from file
        loader = DataSetLoader(regime=regime)

        dataset = SupervisedDataSet(
            features=features,
            labels=labels,
            sensitive_attrs=[],
            num_datapoints=numPoints,
            meta=meta,
        )

        # For each constraint, make a parse tree
        parse_trees = []
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]

            delta = deltas[ii]
            # Create parse tree object
            parse_tree = ParseTree(
                delta=delta, regime=regime, sub_regime=sub_regime, columns=[]
            )

            parse_tree.build_tree(constraint_str=constraint_str)

            parse_trees.append(parse_tree)

        return dataset, model, primary_objective, parse_trees

    return generate_dataset


@pytest.fixture
def gpa_regression_dataset():
    from seldonian.models.models import LinearRegressionModel

    def generate_dataset(constraint_strs, deltas):
        data_pth = "static/datasets/supervised/GPA/gpa_regression_dataset.csv"
        metadata_pth = "static/datasets/supervised/GPA/metadata_regression.json"

        metadata_dict = load_json(metadata_pth)
        regime = metadata_dict["regime"]
        sub_regime = metadata_dict["sub_regime"]
        sensitive_col_names = metadata_dict["sensitive_col_names"]

        regime = "supervised_learning"

        model = LinearRegressionModel()

        # Mean squared error
        primary_objective = objectives.Mean_Squared_Error

        # Load dataset from file
        loader = DataSetLoader(regime=regime)

        dataset = loader.load_supervised_dataset(
            filename=data_pth, metadata_filename=metadata_pth, file_type="csv"
        )

        # For each constraint, make a parse tree
        parse_trees = []
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]

            delta = deltas[ii]
            # Create parse tree object
            parse_tree = ParseTree(
                delta=delta,
                regime="supervised_learning",
                sub_regime="regression",
                columns=sensitive_col_names,
            )

            parse_tree.build_tree(constraint_str=constraint_str)
            parse_trees.append(parse_tree)

        return dataset, model, primary_objective, parse_trees

    return generate_dataset

@pytest.fixture
def gpa_regression_addl_datasets():
    # A fixture for generating a primary dataset and additional datasets
    # for the gpa regression problem
    from seldonian.models.models import LinearRegressionModel

    def generate_datasets(constraint_strs, deltas, batch_size_dict={}):
        """ batch_size_dict is structured d[constraint_str][base_node_name] = int
        """
    
        data_pth = "static/datasets/supervised/GPA/gpa_regression_dataset.csv"
        metadata_pth = "static/datasets/supervised/GPA/metadata_regression.json"

        metadata_dict = load_json(metadata_pth)
        regime = metadata_dict["regime"]
        sub_regime = metadata_dict["sub_regime"]
        all_col_names = metadata_dict["all_col_names"]
        sensitive_col_names = metadata_dict["sensitive_col_names"]

        regime = "supervised_learning"

        model = LinearRegressionModel()

        # Mean squared error
        primary_objective = objectives.Mean_Squared_Error

        # Load dataset from file
        loader = DataSetLoader(regime=regime)

        orig_dataset = loader.load_supervised_dataset(
            filename=data_pth, metadata_filename=metadata_pth, file_type="csv"
        )

        # The new primary dataset has no sensitive attributes
        primary_meta = orig_dataset.meta
        primary_meta.all_col_names = [x for x in primary_meta.all_col_names if x not in primary_meta.sensitive_col_names]
        primary_meta.sensitive_col_names = []
        primary_dataset = SupervisedDataSet(
            features=orig_dataset.features,
            labels=orig_dataset.labels,
            sensitive_attrs=[],
            num_datapoints=orig_dataset.num_datapoints,
            meta=primary_meta
        )

        # Now make a dataset to use for bounding the base nodes
        # Take 80% of the original data
        orig_features = orig_dataset.features
        orig_labels = orig_dataset.labels
        orig_sensitive_attrs = orig_dataset.sensitive_attrs
        num_datapoints_new = int(round(len(orig_features)*0.8))
        rand_indices = np.random.choice(
            a=range(len(orig_features)),
            size=num_datapoints_new,
            replace=False
        )
        new_features = orig_features[rand_indices]
        new_labels = orig_labels[rand_indices]
        new_sensitive_attrs = orig_sensitive_attrs[rand_indices]
        new_meta = SupervisedMetaData(
            sub_regime=sub_regime,
            all_col_names=all_col_names,
            feature_col_names=primary_meta.feature_col_names,
            label_col_names=primary_meta.label_col_names,
            sensitive_col_names=sensitive_col_names,
        )
        new_dataset = SupervisedDataSet(
            features=new_features,
            labels=new_labels,
            sensitive_attrs=new_sensitive_attrs,
            num_datapoints=num_datapoints_new,
            meta=new_meta

        )


        # For each constraint, make a parse tree 
        parse_trees = []
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]
            delta = deltas[ii]
            # Create parse tree object
            parse_tree = ParseTree(
                delta=delta,
                regime="supervised_learning",
                sub_regime="regression",
                columns=sensitive_col_names,
            )

            parse_tree.build_tree(constraint_str=constraint_str)
            parse_trees.append(parse_tree)

        # For each base node in each parse_tree, 
        # add this new dataset to additional_datasets dictionary
        # It is possible that when a parse tree is built, 
        # the constraint string it stores is different than the one that 
        # was used as input. This is because the parser may simplify the expression
        # Therefore, we want to use the constraint string attribute of the built parse 
        # tree as the key to the additional_datasets dict.


        additional_datasets = {}
        for pt in parse_trees:
            additional_datasets[pt.constraint_str] = {}
            base_nodes_this_tree = list(pt.base_node_dict.keys())
            for bn in base_nodes_this_tree:
                additional_datasets[pt.constraint_str][bn] = {
                    "dataset": new_dataset
                }
                try: 
                    additional_datasets[pt.constraint_str][bn]["batch_size"] = batch_size_dict[pt.constraint_str][bn]
                except KeyError:
                    pass

        return primary_dataset, additional_datasets, model, primary_objective, parse_trees

    return generate_datasets


@pytest.fixture
def gpa_classification_dataset():
    from seldonian.models.models import BinaryLogisticRegressionModel

    def generate_dataset(constraint_strs, deltas):
        data_pth = "static/datasets/supervised/GPA/gpa_classification_dataset.csv"
        metadata_pth = "static/datasets/supervised/GPA/metadata_classification.json"

        metadata_dict = load_json(metadata_pth)
        regime = metadata_dict["regime"]
        sub_regime = metadata_dict["sub_regime"]
        columns = metadata_dict["all_col_names"]

        regime = "supervised_learning"

        model = BinaryLogisticRegressionModel()

        # Mean squared error
        primary_objective = objectives.binary_logistic_loss

        # Load dataset from file
        loader = DataSetLoader(regime=regime)

        dataset = loader.load_supervised_dataset(
            filename=data_pth, metadata_filename=metadata_pth, file_type="csv"
        )

        # For each constraint, make a parse tree
        parse_trees = []
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]

            delta = deltas[ii]
            # Create parse tree object
            parse_tree = ParseTree(
                delta=delta,
                regime="supervised_learning",
                sub_regime="classification",
                columns=["M", "F"],
            )

            parse_tree.build_tree(constraint_str=constraint_str)

            parse_trees.append(parse_tree)

        return dataset, model, primary_objective, parse_trees

    return generate_dataset


@pytest.fixture
def gpa_classification_addl_datasets():
    # A fixture for generating a primary dataset and additional datasets
    # for the gpa classification problem
    from seldonian.models.models import BinaryLogisticRegressionModel

    def generate_datasets(constraint_strs, deltas, batch_size_dict={}):
        """ batch_size_dict is structured d[constraint_str][base_node_name] = int
        """
    
        data_pth = "static/datasets/supervised/GPA/gpa_classification_dataset.csv"
        metadata_pth = "static/datasets/supervised/GPA/metadata_classification.json"

        metadata_dict = load_json(metadata_pth)
        all_col_names = metadata_dict["all_col_names"]
        regime = metadata_dict["regime"]
        sub_regime = metadata_dict["sub_regime"]
        sensitive_col_names = metadata_dict["sensitive_col_names"]

        regime = "supervised_learning"

        model = BinaryLogisticRegressionModel()

        # Mean squared error
        primary_objective = objectives.binary_logistic_loss

        # Load dataset from file
        loader = DataSetLoader(regime=regime)

        orig_dataset = loader.load_supervised_dataset(
            filename=data_pth, metadata_filename=metadata_pth, file_type="csv"
        )

        # The new primary dataset has no sensitive attributes
        primary_meta = orig_dataset.meta
        primary_meta.all_col_names = [x for x in primary_meta.all_col_names if x not in primary_meta.sensitive_col_names]
        primary_meta.sensitive_col_names = []
        primary_dataset = SupervisedDataSet(
            features=orig_dataset.features,
            labels=orig_dataset.labels,
            sensitive_attrs=[],
            num_datapoints=orig_dataset.num_datapoints,
            meta=primary_meta
        )

        # Now make a dataset to use for bounding the base nodes
        # Take 80% of the original data
        orig_features = orig_dataset.features
        orig_labels = orig_dataset.labels
        orig_sensitive_attrs = orig_dataset.sensitive_attrs
        num_datapoints_new = int(round(len(orig_features)*0.8))
        rand_indices = np.random.choice(
            a=range(len(orig_features)),
            size=num_datapoints_new,
            replace=False
        )
        new_features = orig_features[rand_indices]
        new_labels = orig_labels[rand_indices]
        new_sensitive_attrs = orig_sensitive_attrs[rand_indices]
        new_meta = SupervisedMetaData(
            sub_regime=sub_regime,
            all_col_names=all_col_names,
            feature_col_names=primary_meta.feature_col_names,
            label_col_names=primary_meta.label_col_names,
            sensitive_col_names=sensitive_col_names,
        )
        new_dataset = SupervisedDataSet(
            features=new_features,
            labels=new_labels,
            sensitive_attrs=new_sensitive_attrs,
            num_datapoints=num_datapoints_new,
            meta=new_meta
        )


        # For each constraint, make a parse tree 
        parse_trees = []
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]
            delta = deltas[ii]
            # Create parse tree object
            parse_tree = ParseTree(
                delta=delta,
                regime="supervised_learning",
                sub_regime=sub_regime,
                columns=sensitive_col_names,
            )

            parse_tree.build_tree(constraint_str=constraint_str)
            parse_trees.append(parse_tree)

        # For each base node in each parse_tree, 
        # add this new dataset to additional_datasets dictionary
        # It is possible that when a parse tree is built, 
        # the constraint string it stores is different than the one that 
        # was used as input. This is because the parser may simplify the expression
        # Therefore, we want to use the constraint string attribute of the built parse 
        # tree as the key to the additional_datasets dict.


        additional_datasets = {}
        for pt in parse_trees:
            additional_datasets[pt.constraint_str] = {}
            base_nodes_this_tree = list(pt.base_node_dict.keys())
            for bn in base_nodes_this_tree:
                additional_datasets[pt.constraint_str][bn] = {
                    "dataset": new_dataset
                }
                try: 
                    additional_datasets[pt.constraint_str][bn]["batch_size"] = batch_size_dict[pt.constraint_str][bn]
                except KeyError:
                    pass

        return primary_dataset, additional_datasets, model, primary_objective, parse_trees

    return generate_datasets


@pytest.fixture
def gpa_multiclass_dataset():
    from seldonian.models.models import BinaryLogisticRegressionModel

    def generate_dataset(constraint_strs, deltas):
        data_pth = "static/datasets/supervised/GPA/gpa_multiclass_dataset.csv"
        metadata_pth = "static/datasets/supervised/GPA/metadata_multiclass.json"

        metadata_dict = load_json(metadata_pth)
        regime = metadata_dict["regime"]
        sub_regime = metadata_dict["sub_regime"]
        sensitive_col_names = metadata_dict["sensitive_col_names"]

        regime = "supervised_learning"

        model = MultiClassLogisticRegressionModel()

        # Mean squared error
        primary_objective = objectives.multiclass_logistic_loss

        # Load dataset from file
        loader = DataSetLoader(regime=regime)

        dataset = loader.load_supervised_dataset(
            filename=data_pth, metadata_filename=metadata_pth, file_type="csv"
        )

        # For each constraint, make a parse tree
        parse_trees = []
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]

            delta = deltas[ii]
            # Create parse tree object
            parse_tree = ParseTree(
                delta=delta,
                regime=regime,
                sub_regime=sub_regime,
                columns=sensitive_col_names,
            )

            parse_tree.build_tree(constraint_str=constraint_str)
            parse_trees.append(parse_tree)

        return dataset, model, primary_objective, parse_trees

    return generate_dataset


@pytest.fixture
def RL_gridworld_dataset():
    from seldonian.RL.environments import gridworld
    from seldonian.RL.RL_model import RL_model
    from seldonian.RL.Agents.Policies.Softmax import DiscreteSoftmax
    from seldonian.RL.Env_Description import Spaces, Env_Description

    def generate_dataset():
        np.random.seed(0)

        # Load data from file into dataset
        data_pth = "static/datasets/RL/gridworld/gridworld_100episodes.pkl"
        metadata_pth = "static/datasets/RL/gridworld/gridworld_metadata.json"

        loader = DataSetLoader(regime="reinforcement_learning")

        dataset = loader.load_RL_dataset_from_episode_file(filename=data_pth)

        # Env description
        num_states = 9  # 3x3 gridworld
        observation_space = Spaces.Discrete_Space(0, num_states - 1)
        action_space = Spaces.Discrete_Space(0, 3)
        env_description = Env_Description.Env_Description(
            observation_space, action_space
        )
        # RL model. setting dict not needed for discrete observation and action space
        policy = DiscreteSoftmax(
            env_description=env_description, hyperparam_and_setting_dict={}
        )

        env_kwargs = {"gamma": 0.9}

        primary_objective = objectives.IS_estimate

        return dataset, policy, env_kwargs, primary_objective

    return generate_dataset


@pytest.fixture
def RL_gridworld_addl_dataset():
    from seldonian.RL.environments import gridworld
    from seldonian.RL.RL_model import RL_model
    from seldonian.RL.Agents.Policies.Softmax import DiscreteSoftmax
    from seldonian.RL.Env_Description import Spaces, Env_Description

    def generate_dataset(constraint_strs,deltas,batch_size_dict={}):
        np.random.seed(0)

        # Load data from file into dataset
        data_pth = "static/datasets/RL/gridworld/gridworld_100episodes.pkl"
        metadata_pth = "static/datasets/RL/gridworld/gridworld_metadata.json"

        loader = DataSetLoader(regime="reinforcement_learning")

        primary_dataset = loader.load_RL_dataset_from_episode_file(filename=data_pth)

        # Make a new dataset which has only the last 50 episodes
        orig_episodes = primary_dataset.episodes
        orig_meta = primary_dataset.meta
        new_dataset = RLDataSet(
            episodes=orig_episodes[-50:],
            meta=orig_meta
            )

        # Env description
        num_states = 9  # 3x3 gridworld
        observation_space = Spaces.Discrete_Space(0, num_states - 1)
        action_space = Spaces.Discrete_Space(0, 3)
        env_description = Env_Description.Env_Description(
            observation_space, action_space
        )
        # RL model. setting dict not needed for discrete observation and action space
        policy = DiscreteSoftmax(
            env_description=env_description, hyperparam_and_setting_dict={}
        )

        env_kwargs = {"gamma": 0.9}
        model = RL_model(policy=policy, env_kwargs=env_kwargs)

        primary_objective = objectives.IS_estimate

        # For each constraint, make a parse tree 
        parse_trees = []
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]
            delta = deltas[ii]
            # Create parse tree object
            parse_tree = ParseTree(
                delta=delta,
                regime="reinforcement_learning",
                sub_regime="all",
                columns=[],
            )

            parse_tree.build_tree(constraint_str=constraint_str)
            parse_trees.append(parse_tree)

        # For each base node in each parse_tree, 
        # add this new dataset to additional_datasets dictionary
        # It is possible that when a parse tree is built, 
        # the constraint string it stores is different than the one that 
        # was used as input. This is because the parser may simplify the expression
        # Therefore, we want to use the constraint string attribute of the built parse 
        # tree as the key to the additional_datasets dict.


        additional_datasets = {}
        for pt in parse_trees:
            additional_datasets[pt.constraint_str] = {}
            base_nodes_this_tree = list(pt.base_node_dict.keys())
            for bn in base_nodes_this_tree:
                additional_datasets[pt.constraint_str][bn] = {
                    "dataset": new_dataset
                }
                try: 
                    additional_datasets[pt.constraint_str][bn]["batch_size"] = batch_size_dict[pt.constraint_str][bn]
                except KeyError:
                    pass

        return primary_dataset, additional_datasets, model, primary_objective, parse_trees

    return generate_dataset


@pytest.fixture
def N_step_mountaincar_dataset():
    from seldonian.RL.environments import n_step_mountaincar
    from seldonian.RL.RL_model import RL_model
    from seldonian.RL.Agents.Policies.Softmax import Softmax
    from seldonian.RL.Env_Description import Spaces, Env_Description

    def generate_dataset():
        np.random.seed(0)

        # Load data from file into dataset
        data_pth = (
            "static/datasets/RL/n_step_mountaincar/n_step_mountaincar_100episodes.pkl"
        )
        metadata_pth = (
            "static/datasets/RL/n_step_mountaincar/n_step_mountaincar_metadata.json"
        )

        loader = DataSetLoader(regime="reinforcement_learning")

        dataset = loader.load_RL_dataset_from_episode_file(filename=data_pth)

        # Env description
        state_space_bounds = np.array([[-1.2, 0.5], [-0.07, 0.07]])
        state_space = Spaces.Continuous_Space(state_space_bounds)
        action_space = Spaces.Discrete_Space(-1, 1)
        env_description = Env_Description.Env_Description(state_space, action_space)

        # RL model. Setting dict is needed since we have a basis
        hyperparam_and_setting_dict = {}
        hyperparam_and_setting_dict["env"] = "n_step_mountaincar"
        hyperparam_and_setting_dict[
            "agent"
        ] = "Parameterized_non_learning_softmax_agent"
        hyperparam_and_setting_dict["basis"] = "Fourier"
        hyperparam_and_setting_dict["order"] = 2
        hyperparam_and_setting_dict["max_coupled_vars"] = -1
        policy = Softmax(
            env_description=env_description,
            hyperparam_and_setting_dict=hyperparam_and_setting_dict,
        )

        env_kwargs = {"gamma": 1.0}

        primary_objective = objectives.IS_estimate

        return dataset, policy, env_kwargs, primary_objective

    return generate_dataset


@pytest.fixture
def RL_gridworld_dataset_alt_rewards():
    from seldonian.RL.environments import gridworld
    from seldonian.RL.RL_model import RL_model
    from seldonian.RL.Agents.Policies.Softmax import DiscreteSoftmax
    from seldonian.RL.Env_Description import Spaces, Env_Description

    def generate_dataset():
        np.random.seed(0)

        # Load data from file into dataset
        data_pth = "static/datasets/RL/gridworld/gridworld_100episodes_2altrewards.pkl"
        metadata_pth = (
            "static/datasets/RL/gridworld/gridworld_2altrewards_metadata.json"
        )

        loader = DataSetLoader(regime="reinforcement_learning")

        dataset = loader.load_RL_dataset_from_episode_file(filename=data_pth)

        # Env description
        num_states = 9  # 3x3 gridworld
        observation_space = Spaces.Discrete_Space(0, num_states - 1)
        action_space = Spaces.Discrete_Space(0, 3)
        env_description = Env_Description.Env_Description(
            observation_space, action_space
        )
        # RL model. setting dict not needed for discrete observation and action space
        policy = DiscreteSoftmax(
            env_description=env_description, hyperparam_and_setting_dict={}
        )

        env_kwargs = {"gamma": 0.9}

        primary_objective = objectives.IS_estimate
        return dataset,policy,env_kwargs,primary_objective
    return generate_dataset

@pytest.fixture
def custom_loan_dataset():
    def generate_dataset():
        # Load German credit dataset but as a custom dataset instead of supervised dataset
        metadata_pth = "static/datasets/custom/german_credit/metadata_german_loan.json"
        meta = load_custom_metadata(metadata_pth)
        
        all_col_names = meta.all_col_names

        # Load data from csv
        data_pth = "static/datasets/custom/german_credit/german_loan_numeric_forseldonian.csv"
        df = pd.read_csv(data_pth, header=None, names=meta.all_col_names)

        sensitive_attrs = df.loc[:, meta.sensitive_col_names].values
        # data is everything besides sensitive attrs (includes labels in this case). 
        # Will handle separating features and labels inside objective functions and measure functions, but not here.
        data_col_names = [col for col in meta.all_col_names if col not in meta.sensitive_col_names]
        data = df.loc[:,data_col_names].values

        num_datapoints = len(data)

        dataset = CustomDataSet(
            data=data,
            sensitive_attrs=sensitive_attrs,
            num_datapoints=num_datapoints,
            meta=meta
        )
        
        return dataset

    return generate_dataset

@pytest.fixture
def custom_loan_addl_dataset():
    def generate_dataset():
        data_pth = "static/datasets/custom/german_credit/german_loan_numeric_forseldonian.csv"
        metadata_pth = "static/datasets/custom/german_credit/metadata_german_loan.json"
        save_dir = '.'
        os.makedirs(save_dir,exist_ok=True)
        # Create dataset from data and metadata file
        regime='custom'
        sub_regime=None

        meta = load_custom_metadata(metadata_pth)

        # One needs to load their custom dataset using their own script
        df = pd.read_csv(data_pth, header=None, names=meta.all_col_names)
        
        sensitive_col_names = meta.sensitive_col_names
        sensitive_attrs = df.loc[:, sensitive_col_names].values
        # data is everything else (includes labels in this case). 
        # will handle separating features and labels inside objective functions and measure functions
        data_col_names = [col for col in meta.all_col_names if col not in sensitive_col_names]
        data = df.loc[:,data_col_names].values

        num_datapoints = len(data)

        primary_dataset = CustomDataSet(
            data=data,
            sensitive_attrs=[],
            num_datapoints=num_datapoints,
            meta=meta
        )

        new_num_datapoints = 500
        new_data = data[0:new_num_datapoints]
        new_sensitive_attrs = sensitive_attrs[0:new_num_datapoints]

        new_dataset = CustomDataSet(
            data=new_data,
            sensitive_attrs=new_sensitive_attrs,
            num_datapoints=new_num_datapoints,
            meta=meta
            )
       

        # Use logistic regression model
        model = BinaryLogisticRegressionModel()
        
        # Define the primary objective to be log loss
        # Can just call the existing log loss function
        # but must wrap it because in this custom 
        # setting we don't know what features and labels
        # are a priori. We just have a "data" argument 
        # that we have to manipulate accordingly.


        def custom_log_loss(model,theta,data,**kwargs):
            """Calculate average logistic loss
            over all data points for binary classification.

            :param model: SeldonianModel instance
            :param theta: The parameter weights
            :type theta: numpy ndarray
            :param data: A list of samples, where in this case samples are
                rows of a 2D numpy array

            :return: mean logistic loss
            :rtype: float
            """
            # Figure out features and labels
            # In this case I know that the label column is the final column
            # I also know that data is a 2D numpy array. The data structure
            # will be custom to the use case, so user will have to manipulate 
            # accordingly. 
            features = data[:,:-1]
            labels = data[:,-1]
            return objectives.binary_logistic_loss(model, theta, features, labels, **kwargs)

        primary_objective = custom_log_loss
        # Define behavioral constraints
        epsilon = 0.6
        constraint_strs = [f'min((CPR | [M])/(CPR | [F]),(CPR | [F])/(CPR | [M])) >= {epsilon}'] 
        deltas = [0.05]
        
        # Define custom measure function for CPR and register it when making parse tree
        def custom_vector_Positive_Rate(model, theta, data, **kwargs):
            """
            Calculate positive rate
            for each observation. Meaning depends on whether
            binary or multi-class classification.

            :param model: SeldonianModel instance
            :param theta: The parameter weights
            :type theta: numpy ndarray
            :param data: A list of samples, where in this case samples are
                rows of a 2D numpy array

            :return: Positive rate for each observation
            :rtype: numpy ndarray(float between 0 and 1)
            """
            features = data[:,:-1]
            labels = data[:,-1]
            return zhat_funcs._vector_Positive_Rate_binary(model, theta, features, labels)

        custom_measure_functions = {
            "CPR": custom_vector_Positive_Rate
        }
        # For each constraint (in this case only one), make a parse tree
        parse_trees = []
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]

            delta = deltas[ii]

            # Create parse tree object
            pt = ParseTree(
                delta=delta, regime=regime, sub_regime=sub_regime, columns=sensitive_col_names,
                custom_measure_functions=custom_measure_functions
            )

            # Fill out tree
            pt.build_tree(
                constraint_str=constraint_str
            )

            parse_trees.append(pt)

        # For each base node in each parse_tree, 
        # add this new dataset to additional_datasets dictionary
        # It is possible that when a parse tree is built, 
        # the constraint string it stores is different than the one that 
        # was used as input. This is because the parser may simplify the expression
        # Therefore, we want to use the constraint string attribute of the built parse 
        # tree as the key to the additional_datasets dict.

        additional_datasets = {}
        for pt in parse_trees:
            additional_datasets[pt.constraint_str] = {}
            base_nodes_this_tree = list(pt.base_node_dict.keys())
            for bn in base_nodes_this_tree:
                additional_datasets[pt.constraint_str][bn] = {
                    "dataset": new_dataset
                }

        return primary_dataset, additional_datasets, model, primary_objective, parse_trees


    return generate_dataset


@pytest.fixture
def custom_text_spec():
    import seldonian.models.custom_text_model as custom_text_model

    def generate_spec():
        # Load some string data in as lists of lists
        N_chars = 100
        l=[chr(x) for x in np.random.randint(97,122,N_chars)] # lowercase letters
        data = [l[i*3:i*3+3] for i in range(N_chars//3)]

        all_col_names = ["string"]
        meta = CustomMetaData(all_col_names=all_col_names)
        dataset = CustomDataSet(data=data, sensitive_attrs=[], num_datapoints=len(data), meta=meta)

        regime='custom'
        sub_regime=None
        sensitive_attrs = []

        num_datapoints = len(data)

        dataset = CustomDataSet(
            data=data,
            sensitive_attrs=sensitive_attrs,
            num_datapoints=num_datapoints,
            meta=meta
        )
        frac_data_in_safety=0.6
        sensitive_col_names = []

        model = custom_text_model.CustomTextModel()

        def custom_initial_solution_fn(model,data,**kwargs):
            return np.array([-1.0,0.0,1.0])

        def custom_loss_fn(model,theta,data,**kwargs):
            """Calculate average logistic loss
            over all data points for binary classification.

            :param model: SeldonianModel instance
            :param theta: The parameter weights
            :type theta: numpy ndarray
            :param data: A list of samples, where in this case samples are
                lists of length three with each element a single character

            :return: mean of the predictions
            :rtype: float
            """
            # Figure out features and labels
            # In this case I know that the label column is the final column
            # I also know that data is a 2D numpy array. The data structure
            # will be custom to the use case, so user will have to manipulate 
            # accordingly. 
            predictions = model.predict(theta,data) # floats length of data
            loss = np.mean(predictions)
            return loss


        # Define behavioral constraint
        constraint_str = 'CUST_LOSS <= 30.0'
        delta = 0.05
        
        # Define custom measure function for CPR and register it when making parse tree
        def custom_measure_function(model, theta, data, **kwargs):
            """
            Calculate 
            for each observation. Meaning depends on whether
            binary or multi-class classification.

            :param model: SeldonianModel instance
            :param theta: The parameter weights
            :type theta: numpy ndarray
            :param data: A list of samples, where in this case samples are
                lists of length three with each element a single character

            :return: Positive rate for each observation
            :rtype: numpy ndarray(float between 0 and 1)
            """
            predictions = model.predict(theta,data)
            return predictions

        custom_measure_functions = {
            "CUST_LOSS": custom_measure_function
        }
        
        # Create parse tree object
        pt = ParseTree(
            delta=delta, regime=regime, sub_regime=sub_regime, columns=sensitive_col_names,
            custom_measure_functions=custom_measure_functions
        )

        # Fill out tree
        pt.build_tree(
            constraint_str=constraint_str
        )

        parse_trees = [pt]

        # Use vanilla Spec object for custom datasets.
        spec = Spec(
            dataset=dataset,
            model=model,
            parse_trees=parse_trees,
            frac_data_in_safety=frac_data_in_safety,
            primary_objective=custom_loss_fn,
            initial_solution_fn=custom_initial_solution_fn,
            use_builtin_primary_gradient_fn=False,
            optimization_technique='gradient_descent',
            optimizer='adam',
            optimization_hyperparams={
                'lambda_init'   : np.array([0.5]),
                'alpha_theta'   : 0.01,
                'alpha_lamb'    : 0.01,
                'beta_velocity' : 0.9,
                'beta_rmsprop'  : 0.95,
                'use_batches'   : False,
                'num_iters'     : 100,
                'gradient_library': "autograd",
                'hyper_search'  : None,
                'verbose'       : True,
            }
        )

        return spec

    return generate_spec

@pytest.fixture
def custom_loan_spec():
    def generate_spec():
        # Load some string data in as lists of lists
        data_pth = "static/datasets/custom/german_credit/german_loan_numeric_forseldonian.csv"
        metadata_pth = "static/datasets/custom/german_credit/metadata_german_loan.json"
        save_dir = '.'
        os.makedirs(save_dir,exist_ok=True)
        # Create dataset from data and metadata file
        regime='custom'
        sub_regime=None

        meta = load_custom_metadata(metadata_pth)

        # One needs to load their custom dataset using their own script
        df = pd.read_csv(data_pth, header=None, names=meta.all_col_names)

        sensitive_attrs = df.loc[:, meta.sensitive_col_names].values
        # data is everything else (includes labels in this case). 
        # will handle separating features and labels inside objective functions and measure functions
        data_col_names = [col for col in meta.all_col_names if col not in meta.sensitive_col_names]
        data = df.loc[:,data_col_names].values

        num_datapoints = len(data)

        dataset = CustomDataSet(
            data=data,
            sensitive_attrs=sensitive_attrs,
            num_datapoints=num_datapoints,
            meta=meta
        )
       
        sensitive_col_names = dataset.meta.sensitive_col_names

        # Use logistic regression model
        model = BinaryLogisticRegressionModel()
        
        # Define the primary objective to be log loss
        # Can just call the existing log loss function
        # but must wrap it because in this custom 
        # setting we don't know what features and labels
        # are a priori. We just have a "data" argument 
        # that we have to manipulate accordingly.

        def custom_initial_solution_fn(model,data,**kwargs):
            features = data[:,:-1]
            labels = data[:,-1]
            return model.fit(features,labels)

        def custom_log_loss(model,theta,data,**kwargs):
            """Calculate average logistic loss
            over all data points for binary classification.

            :param model: SeldonianModel instance
            :param theta: The parameter weights
            :type theta: numpy ndarray
            :param data: A list of samples, where in this case samples are
                rows of a 2D numpy array

            :return: mean logistic loss
            :rtype: float
            """
            # Figure out features and labels
            # In this case I know that the label column is the final column
            # I also know that data is a 2D numpy array. The data structure
            # will be custom to the use case, so user will have to manipulate 
            # accordingly. 
            features = data[:,:-1]
            labels = data[:,-1]
            return objectives.binary_logistic_loss(model, theta, features, labels, **kwargs)

        # Define behavioral constraints
        epsilon = 0.6
        constraint_name = "disparate_impact"
        if constraint_name == "disparate_impact":
            constraint_strs = [f'min((CPR | [M])/(CPR | [F]),(CPR | [F])/(CPR | [M])) >= {epsilon}'] 
        deltas = [0.05]
        
        # Define custom measure function for CPR and register it when making parse tree
        def custom_vector_Positive_Rate(model, theta, data, **kwargs):
            """
            Calculate positive rate
            for each observation. Meaning depends on whether
            binary or multi-class classification.

            :param model: SeldonianModel instance
            :param theta: The parameter weights
            :type theta: numpy ndarray
            :param data: A list of samples, where in this case samples are
                rows of a 2D numpy array

            :return: Positive rate for each observation
            :rtype: numpy ndarray(float between 0 and 1)
            """
            features = data[:,:-1]
            labels = data[:,-1]
            return zhat_funcs._vector_Positive_Rate_binary(model, theta, features, labels)

        custom_measure_functions = {
            "CPR": custom_vector_Positive_Rate
        }
        # For each constraint (in this case only one), make a parse tree
        parse_trees = []
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]

            delta = deltas[ii]

            # Create parse tree object
            pt = ParseTree(
                delta=delta, regime=regime, sub_regime=sub_regime, columns=sensitive_col_names,
                custom_measure_functions=custom_measure_functions
            )

            # Fill out tree
            pt.build_tree(
                constraint_str=constraint_str
            )

            parse_trees.append(pt)


        spec = Spec(
            dataset=dataset,
            model=model,
            parse_trees=parse_trees,
            frac_data_in_safety=0.6,
            primary_objective=custom_log_loss,
            initial_solution_fn=custom_initial_solution_fn,
            use_builtin_primary_gradient_fn=False,
            optimization_technique='gradient_descent',
            optimizer='adam',
            optimization_hyperparams={
                'lambda_init'   : np.array([0.5]),
                'alpha_theta'   : 0.01,
                'alpha_lamb'    : 0.01,
                'beta_velocity' : 0.9,
                'beta_rmsprop'  : 0.95,
                'use_batches'   : True,
                'batch_size'    : 50,
                'n_epochs'      : 2,
                'gradient_library': "autograd",
                'hyper_search'  : None,
                'verbose'       : True,
            }
        )
        return spec

    return generate_spec


#### Hyperparamater fixtures

@pytest.fixture
def Hyperparam_spec(all_frac_data_in_safety, n_bootstrap_trials=100, n_bootstrap_workers=30,
        use_bs_pools=False, hyper_schema=None):
    if hyper_schema is None:
        hyper_schema = HyperSchema(
                {
                    "frac_data_in_safety": {
                        "values": all_frac_data_in_safety,
                        "hyper_type": "SA"
                        },
                    "alpha_theta": {
                        "values": [0.001, 0.005, 0.05],
                        "hyper_type": "optimization"
                        },
                    "num_iters": {
                        "values": [500,1000],
                        "hyper_type": "optimization"
                        },
                    "bound_inflation_factor": {
                        "values": [1.0, 2.0, 3.0],
                        "hyper_type": "SA"
                        }
                    }
                )

    HS_spec = HyperparameterSelectionSpec(
            hyper_schema=hyper_schema,
            n_bootstrap_trials=n_bootstrap_trials,
            n_bootstrap_workers=n_bootstrap_workers,
            use_bs_pools=use_bs_pools,
            confidence_interval_type=None
    )

    return HS_spec
