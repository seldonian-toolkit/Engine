import os
import shutil
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pytest

from seldonian.parse_tree.parse_tree import *
from seldonian.utils.io_utils import (load_json,
    load_pickle)
from seldonian.utils.tutorial_utils import generate_data
from seldonian.dataset import (DataSetLoader,
    RLDataSet,SupervisedDataSet)
from seldonian.spec import SupervisedSpec
from seldonian.models import objectives 
from seldonian.models.models import *


@pytest.fixture
def stump():
    def stump_function(operator_type,left_bounds,right_bounds,
        regime='supervised_learning',sub_regime='classification'):
        # A parse tree with a root node and left and right children only
        root = InternalNode(operator_type)
        root.index=0
        root.left = BaseNode('a')
        root.left.index=1
        root.right = BaseNode('b')
        root.right.index=2

        pt = ParseTree(delta=0.05,regime=regime,sub_regime=sub_regime)
        # pt.create_from_ast("a+b")
        pt.root = root
        pt.root.left.lower  = left_bounds[0]
        pt.root.left.upper  = left_bounds[1]
        pt.root.right.lower = right_bounds[0]
        pt.root.right.upper = right_bounds[1]
        pt.n_nodes = 3
        pt.n_base_nodes = 2
        pt.base_node_dict = {
            'a':{
                'bound_method':'manual',
                'bound_computed':False,
                'lower':float("-inf"),
                'upper':float("inf"),
                'data':None,
                'datasize':0
                },
            'b':{
                'bound_method':'manual',
                'bound_computed':False,
                'lower':float("-inf"),
                'upper':float("inf"),
                'data':None,
                'datasize':0
                },
        }
        return pt
    return stump_function

@pytest.fixture
def edge():
    def edge_function(operator_type,left_bounds,
        regime='supervised_learning',sub_regime='classification'):
        # A parse tree with a single edge
        assert operator_type in ['abs','exp']
        root = InternalNode(operator_type)
        root.left = BaseNode('a')
        pt = ParseTree(delta=0.05,regime=regime,sub_regime=sub_regime)
        pt.root = root
        pt.root.left.lower  = left_bounds[0]
        pt.root.left.upper  = left_bounds[1]
        pt.n_nodes = 2
        pt.n_base_nodes = 1
        pt.base_node_dict = {
            'a':{
                'bound_method':'manual',
                'bound_computed':False,
                'lower':float("-inf"),
                'upper':float("inf"),
                'data':None,
                'datasize':0
                },
        }
        return pt
    return edge_function

@pytest.fixture
def spec_garbage_collector():
    save_dir = "./tests/specfiles"
    """ Fixture to create and then remove results_dir and any files it may contain"""
    print("----------- Setup spec_garbage_collector -----------")
    os.makedirs(save_dir,exist_ok=True)
    yield
    print("----------- Teardown spec_garbage_collector -----------")
    shutil.rmtree(save_dir)

@pytest.fixture
def simulated_regression_dataset_aslists():
    """ Generate features, labels and sensitive attributes as lists """
    from seldonian.models.models import LinearRegressionModel
    def generate_dataset(constraint_strs,deltas,numPoints=1000):

        regime='supervised_learning'
        sub_regime='regression'
        np.random.seed(0)

        model = LinearRegressionModelListFeatures() # we don't have a model that supports lists of 
        # features/arrays yet, but one could create one.
        X1,Y = generate_data(
            numPoints,loc_X=0.0,loc_Y=0.0,sigma_X=1.0,sigma_Y=1.0)
        X2 = X1**2
        meta_information = {}
        meta_information['feature_col_names'] = ['feature1']
        meta_information['label_col_names'] = ['label']
        meta_information['sensitive_col_names'] = []

        # 3. Make a dataset object
        features = [np.expand_dims(X1,axis=1),np.expand_dims(X2,axis=1)]
        labels = Y
        sensitive_attrs=list()

        # Mean squared error
        primary_objective = objectives.Mean_Squared_Error

        # Load dataset from file
        loader = DataSetLoader(
            regime=regime)

        dataset = SupervisedDataSet(
            features=features,
            labels=labels,
            sensitive_attrs=sensitive_attrs,
            num_datapoints=numPoints,
            meta_information=meta_information)

        # For each constraint, make a parse tree
        parse_trees = []
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]

            delta = deltas[ii]
            # Create parse tree object
            parse_tree = ParseTree(delta=delta,
                regime=regime,sub_regime=sub_regime,
                columns=[])

            # Fill out tree
            parse_tree.create_from_ast(constraint_str)
            # assign deltas for each base node
            # use equal weighting for each base node
            parse_tree.assign_deltas(weight_method='equal')

            # Assign bounds needed on the base nodes
            parse_tree.assign_bounds_needed()
            
            parse_trees.append(parse_tree)

        return dataset,model,primary_objective,parse_trees
    
    return generate_dataset


@pytest.fixture
def simulated_regression_dataset():

    from seldonian.models.models import LinearRegressionModel
    def generate_dataset(constraint_strs,deltas,numPoints=1000):

        regime='supervised_learning'
        sub_regime='regression'
        np.random.seed(0)

        model = LinearRegressionModel()
        X,Y = generate_data(
            numPoints,loc_X=0.0,loc_Y=0.0,sigma_X=1.0,sigma_Y=1.0)
        
        meta_information = {}
        meta_information['feature_col_names'] = ['feature1']
        meta_information['label_col_names'] = ['label']
        meta_information['sensitive_col_names'] = []

        # 3. Make a dataset object
        features = np.expand_dims(X,axis=1)
        labels = Y

        # Mean squared error
        primary_objective = objectives.Mean_Squared_Error

        # Load dataset from file
        loader = DataSetLoader(
            regime=regime)

        dataset = SupervisedDataSet(
            features=features,
            labels=labels,
            sensitive_attrs=[],
            num_datapoints=numPoints,
            meta_information=meta_information)

        # For each constraint, make a parse tree
        parse_trees = []
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]

            delta = deltas[ii]
            # Create parse tree object
            parse_tree = ParseTree(delta=delta,
                regime=regime,sub_regime=sub_regime,
                columns=[])

            # Fill out tree
            parse_tree.create_from_ast(constraint_str)
            # assign deltas for each base node
            # use equal weighting for each base node
            parse_tree.assign_deltas(weight_method='equal')

            # Assign bounds needed on the base nodes
            parse_tree.assign_bounds_needed()
            
            parse_trees.append(parse_tree)

        return dataset,model,primary_objective,parse_trees
    
    return generate_dataset

@pytest.fixture
def gpa_regression_dataset():

    from seldonian.models.models import LinearRegressionModel
    def generate_dataset(constraint_strs,deltas,):
        data_pth = 'static/datasets/supervised/GPA/gpa_regression_dataset.csv'
        metadata_pth = 'static/datasets/supervised/GPA/metadata_regression.json'

        metadata_dict = load_json(metadata_pth)
        regime = metadata_dict['regime']
        sub_regime = metadata_dict['sub_regime']
        sensitive_col_names = metadata_dict['sensitive_col_names']
                    
        regime='supervised_learning'

        model = LinearRegressionModel()

        # Mean squared error
        primary_objective = objectives.Mean_Squared_Error

        # Load dataset from file
        loader = DataSetLoader(
            regime=regime)

        dataset = loader.load_supervised_dataset(
            filename=data_pth,
            metadata_filename=metadata_pth,
            file_type='csv')

        # For each constraint, make a parse tree
        parse_trees = []
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]

            delta = deltas[ii]
            # Create parse tree object
            parse_tree = ParseTree(delta=delta,
                regime='supervised_learning',sub_regime='regression',
                columns=sensitive_col_names)

            # Fill out tree
            parse_tree.create_from_ast(constraint_str)
            # assign deltas for each base node
            # use equal weighting for each base node
            parse_tree.assign_deltas(weight_method='equal')

            # Assign bounds needed on the base nodes
            parse_tree.assign_bounds_needed()
            
            parse_trees.append(parse_tree)

        return dataset,model,primary_objective,parse_trees
    
    return generate_dataset

@pytest.fixture
def gpa_classification_dataset():

    from seldonian.models.models import BinaryLogisticRegressionModel
    def generate_dataset(constraint_strs,deltas):

        data_pth = 'static/datasets/supervised/GPA/gpa_classification_dataset.csv'
        metadata_pth = 'static/datasets/supervised/GPA/metadata_classification.json'

        metadata_dict = load_json(metadata_pth)
        regime = metadata_dict['regime']
        sub_regime = metadata_dict['sub_regime']
        columns = metadata_dict['all_col_names']
                    
        regime='supervised_learning'

        model = BinaryLogisticRegressionModel()

        # Mean squared error
        primary_objective = objectives.binary_logistic_loss

        # Load dataset from file
        loader = DataSetLoader(
            regime=regime)

        dataset = loader.load_supervised_dataset(
            filename=data_pth,
            metadata_filename=metadata_pth,
            file_type='csv')

        # For each constraint, make a parse tree
        parse_trees = []
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]

            delta = deltas[ii]
            # Create parse tree object
            parse_tree = ParseTree(delta=delta,
                regime='supervised_learning',sub_regime='classification',
                columns=["M","F"])

            # Fill out tree
            parse_tree.create_from_ast(constraint_str)
            # assign deltas for each base node
            # use equal weighting for each base node
            parse_tree.assign_deltas(weight_method='equal')

            # Assign bounds needed on the base nodes
            parse_tree.assign_bounds_needed()
            
            parse_trees.append(parse_tree)

        return dataset,model,primary_objective,parse_trees
    
    return generate_dataset

@pytest.fixture
def gpa_multiclass_dataset():

    from seldonian.models.models import BinaryLogisticRegressionModel
    def generate_dataset(constraint_strs,deltas):

        data_pth = 'static/datasets/supervised/GPA/gpa_multiclass_dataset.csv'
        metadata_pth = 'static/datasets/supervised/GPA/metadata_multiclass.json'

        metadata_dict = load_json(metadata_pth)
        regime = metadata_dict['regime']
        sub_regime = metadata_dict['sub_regime']
        sensitive_col_names = metadata_dict['sensitive_col_names']
                    
        regime='supervised_learning'

        model = MultiClassLogisticRegressionModel()

        # Mean squared error
        primary_objective = objectives.multiclass_logistic_loss

        # Load dataset from file
        loader = DataSetLoader(
            regime=regime)

        dataset = loader.load_supervised_dataset(
            filename=data_pth,
            metadata_filename=metadata_pth,
            file_type='csv')

        # For each constraint, make a parse tree
        parse_trees = []
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]

            delta = deltas[ii]
            # Create parse tree object
            parse_tree = ParseTree(delta=delta,
                regime=regime,sub_regime=sub_regime,
                columns=sensitive_col_names)

            # Fill out tree
            parse_tree.create_from_ast(constraint_str)
            # assign deltas for each base node
            # use equal weighting for each base node
            parse_tree.assign_deltas(weight_method='equal')

            # Assign bounds needed on the base nodes
            parse_tree.assign_bounds_needed()
            
            parse_trees.append(parse_tree)

        return dataset,model,primary_objective,parse_trees
    
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
        data_pth = 'static/datasets/RL/gridworld/gridworld_100episodes.pkl'
        metadata_pth = 'static/datasets/RL/gridworld/gridworld_metadata.json'

        loader = DataSetLoader(
            regime="reinforcement_learning")

        dataset = loader.load_RL_dataset_from_episode_file(
            filename=data_pth)

        # Env description 
        num_states = 9 # 3x3 gridworld
        observation_space = Spaces.Discrete_Space(0, num_states-1)
        action_space = Spaces.Discrete_Space(0, 3)
        env_description = Env_Description.Env_Description(observation_space, action_space)
        # RL model. setting dict not needed for discrete observation and action space
        policy = DiscreteSoftmax(
            env_description=env_description,
            hyperparam_and_setting_dict={}
        )

        env_kwargs={'gamma':0.9}

        primary_objective = objectives.IS_estimate

        return dataset,policy,env_kwargs,primary_objective
    
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
        data_pth = 'static/datasets/RL/n_step_mountaincar/n_step_mountaincar_100episodes.pkl'
        metadata_pth = 'static/datasets/RL/n_step_mountaincar/n_step_mountaincar_metadata.json'

        loader = DataSetLoader(
            regime="reinforcement_learning")

        dataset = loader.load_RL_dataset_from_episode_file(
            filename=data_pth)

        # Env description 
        state_space_bounds = np.array([[-1.2, 0.5], [-.07, .07]])
        state_space = Spaces.Continuous_Space(state_space_bounds)
        action_space = Spaces.Discrete_Space(-1, 1)
        env_description = Env_Description.Env_Description(state_space, action_space)
        
        # RL model. Setting dict is needed since we have a basis
        hyperparam_and_setting_dict = {}
        hyperparam_and_setting_dict["env"] = "n_step_mountaincar"
        hyperparam_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
        hyperparam_and_setting_dict["basis"] = "Fourier"
        hyperparam_and_setting_dict["order"] = 2
        hyperparam_and_setting_dict["max_coupled_vars"] = -1
        policy = Softmax(
            env_description=env_description,
            hyperparam_and_setting_dict=hyperparam_and_setting_dict
        )

        env_kwargs={'gamma':1.0}

        primary_objective = objectives.IS_estimate

        return dataset,policy,env_kwargs,primary_objective
    
    return generate_dataset