import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pytest

from seldonian.parse_tree.parse_tree import *
from seldonian.utils.io_utils import (load_json,
    load_pickle)
from seldonian.dataset import (DataSetLoader,
    RLDataSet)



@pytest.fixture
def stump():
    def stump_function(operator_type,left_bounds,right_bounds,
        regime='supervised',sub_regime='classification'):
        # A parse tree with a root node and left and right children only
        root = InternalNode(operator_type)
        root.left = BaseNode('a')
        root.right = BaseNode('b')
        pt = ParseTree(delta=0.05,regime=regime,sub_regime=sub_regime)
        pt.root = root
        pt.root.left.lower  = left_bounds[0]
        pt.root.left.upper  = left_bounds[1]
        pt.root.right.lower = right_bounds[0]
        pt.root.right.upper = right_bounds[1]
        pt.n_nodes = 3
        pt.n_base_nodes = 2
        pt.base_node_dict = {
            'a':{
                'bound_computed':False,
                'lower':float("-inf"),
                'upper':float("inf"),
                'data':None,
                'datasize':0
                },
            'b':{
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
        regime='supervised',sub_regime='classification'):
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
def generate_data():
    def generate_data_function(numPoints,loc_X=0.0,loc_Y=0.0,sigma_X=1.0,sigma_Y=1.0):
        X =     np.random.normal(loc_X, sigma_X, numPoints) # Sample x from a standard normal distribution
        Y = X + np.random.normal(loc_Y, sigma_Y, numPoints) # Set y to be x, plus noise from a standard normal distribution
        return (X,Y)
    
    return generate_data_function

@pytest.fixture
def gpa_regression_dataset():

    from seldonian.models.model import LinearRegressionModel
    def generate_dataset(constraint_strs,deltas):

        data_pth = 'static/datasets/GPA/gpa_regression_dataset.csv'
        metadata_pth = 'static/datasets/GPA/metadata_regression.json'

        metadata_dict = load_json(metadata_pth)
        regime = metadata_dict['regime']
        sub_regime = metadata_dict['sub_regime']
        columns = metadata_dict['columns']
                    
        include_sensitive_columns = False
        include_intercept_term = True
        regime='supervised'

        model_class = LinearRegressionModel

        # Mean squared error
        primary_objective = model_class().sample_Mean_Squared_Error

        # Load dataset from file
        loader = DataSetLoader(
            regime=regime)

        dataset = loader.load_supervised_dataset(
            filename=data_pth,
            metadata_filename=metadata_pth,
            include_sensitive_columns=include_sensitive_columns,
            include_intercept_term=include_intercept_term,
            file_type='csv')

        # For each constraint, make a parse tree
        parse_trees = []
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]

            delta = deltas[ii]
            # Create parse tree object
            parse_tree = ParseTree(delta=delta,
                regime='supervised',sub_regime='regression')

            # Fill out tree
            parse_tree.create_from_ast(constraint_str)
            # assign deltas for each base node
            # use equal weighting for each base node
            parse_tree.assign_deltas(weight_method='equal')

            # Assign bounds needed on the base nodes
            parse_tree.assign_bounds_needed()
            
            parse_trees.append(parse_tree)

        return dataset,model_class,primary_objective,parse_trees
    
    return generate_dataset

@pytest.fixture
def gpa_classification_dataset():

    from seldonian.models.model import LogisticRegressionModel
    def generate_dataset(constraint_strs,deltas):

        data_pth = 'static/datasets/GPA/gpa_classification_dataset.csv'
        metadata_pth = 'static/datasets/GPA/metadata_classification.json'

        metadata_dict = load_json(metadata_pth)
        regime = metadata_dict['regime']
        sub_regime = metadata_dict['sub_regime']
        columns = metadata_dict['columns']
                    
        include_sensitive_columns = False
        include_intercept_term = True
        regime='supervised'

        model_class = LogisticRegressionModel

        # Mean squared error
        primary_objective = model_class().sample_logistic_loss

        # Load dataset from file
        loader = DataSetLoader(
            regime=regime)

        dataset = loader.load_supervised_dataset(
            filename=data_pth,
            metadata_filename=metadata_pth,
            include_sensitive_columns=include_sensitive_columns,
            include_intercept_term=include_intercept_term,
            file_type='csv')

        # For each constraint, make a parse tree
        parse_trees = []
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]

            delta = deltas[ii]
            # Create parse tree object
            parse_tree = ParseTree(delta=delta,
                regime='supervised',sub_regime='classification',
                columns=["M","F"])

            # Fill out tree
            parse_tree.create_from_ast(constraint_str)
            # assign deltas for each base node
            # use equal weighting for each base node
            parse_tree.assign_deltas(weight_method='equal')

            # Assign bounds needed on the base nodes
            parse_tree.assign_bounds_needed()
            
            parse_trees.append(parse_tree)

        return dataset,model_class,primary_objective,parse_trees
    
    return generate_dataset


@pytest.fixture
def RL_gridworld_dataset():
    from seldonian.RL.environments import gridworld3x3
    from seldonian.models.model import TabularSoftmaxModel

    def generate_dataset(constraint_strs,deltas):
        env = gridworld3x3.Environment()
        regime='RL'

        model_class = TabularSoftmaxModel
        model_instance = model_class(env)
        # Mean squared error
        primary_objective = model_instance.sample_IS_estimate

        # Load data into dataset
        data_pth = 'static/datasets/RL/gridworld/gridworld3x3_250episodes.pkl'

        episodes = load_pickle(data_pth)
        columns = ['O','A','R','pi']
        dataset = RLDataSet(episodes=episodes,
            meta_information=columns)

        # For each constraint, make a parse tree
        parse_trees = []
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]

            delta = deltas[ii]
            # Create parse tree object
            parse_tree = ParseTree(delta=delta,
                regime='RL',sub_regime='all')

            # Fill out tree
            parse_tree.create_from_ast(constraint_str)
            # assign deltas for each base node
            # use equal weighting for each base node
            parse_tree.assign_deltas(weight_method='equal')

            # Assign bounds needed on the base nodes
            parse_tree.assign_bounds_needed()
            
            parse_trees.append(parse_tree)


        return dataset,model_class,primary_objective,parse_trees,env
    
    return generate_dataset