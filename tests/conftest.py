import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pytest

from seldonian.parse_tree.parse_tree import *
from seldonian.utils.io_utils import (load_json,
    load_pickle)
from seldonian.dataset import (DataSetLoader,
    RLDataSet,SupervisedDataSet)
from seldonian.spec import SupervisedSpec



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
def generate_data():
    def generate_data_function(numPoints,loc_X=0.0,loc_Y=0.0,sigma_X=1.0,sigma_Y=1.0):
        X =     np.random.normal(loc_X, sigma_X, numPoints) # Sample x from a standard normal distribution
        Y = X + np.random.normal(loc_Y, sigma_Y, numPoints) # Set y to be x, plus noise from a standard normal distribution
        return (X,Y)
    
    return generate_data_function


@pytest.fixture
def synthetic_dataset(generate_data):
    from seldonian.models.models import LinearRegressionModel
    
    def generate_dataset(constraint_strs,deltas,numPoints=1000,include_intercept_term=False):
        rseed=0
        np.random.seed(rseed) 
        columns=['feature1','label']
        model_class = LinearRegressionModel
        X,Y = generate_data(
            numPoints,loc_X=0.0,loc_Y=0.0,sigma_X=1.0,sigma_Y=0.2)
        import matplotlib.pyplot as plt
        rows = np.hstack([np.expand_dims(X,axis=1),np.expand_dims(Y,axis=1)])
        df = pd.DataFrame(rows,columns=columns)

        parse_trees = []
        
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]

            delta = deltas[ii]
            # Create parse tree object
            parse_tree = ParseTree(delta=delta,
                regime='supervised_learning',sub_regime='regression',
                columns=[])

            # Fill out tree
            parse_tree.create_from_ast(constraint_str)
            # assign deltas for each base node
            # use equal weighting for each base node
            parse_tree.assign_deltas(weight_method='equal')

            # Assign bounds needed on the base nodes
            parse_tree.assign_bounds_needed()
            
            parse_trees.append(parse_tree)

        dataset = SupervisedDataSet(df=df,meta_information=columns,
            label_column='label',
            sensitive_column_names=[],
            include_sensitive_columns=False,
            include_intercept_term=include_intercept_term
        )
        frac_data_in_safety=0.6

        # Create spec object
        # Will warn because of initial solution trying to fit with not enough data
        spec = SupervisedSpec(
            dataset=dataset,
            model_class=model_class,
            frac_data_in_safety=frac_data_in_safety,
            primary_objective=model_class().default_objective,
            use_builtin_primary_gradient_fn=True,
            parse_trees=parse_trees,
            initial_solution_fn=model_class().fit,
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
        return spec
    
    return generate_dataset


@pytest.fixture
def gpa_regression_dataset():

    from seldonian.models.models import LinearRegressionModel
    def generate_dataset(constraint_strs,deltas):

        data_pth = 'static/datasets/supervised/GPA/gpa_regression_dataset.csv'
        metadata_pth = 'static/datasets/supervised/GPA/metadata_regression.json'

        metadata_dict = load_json(metadata_pth)
        regime = metadata_dict['regime']
        sub_regime = metadata_dict['sub_regime']
        columns = metadata_dict['columns']
        sensitive_columns = metadata_dict['sensitive_columns']
                    
        include_sensitive_columns = False
        include_intercept_term = True
        regime='supervised_learning'

        model_class = LinearRegressionModel

        # Mean squared error
        primary_objective = model_class().default_objective

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
                regime='supervised_learning',sub_regime='regression',
                columns=sensitive_columns)

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

    from seldonian.models.models import LogisticRegressionModel
    def generate_dataset(constraint_strs,deltas):

        data_pth = 'static/datasets/supervised/GPA/gpa_classification_dataset.csv'
        metadata_pth = 'static/datasets/supervised/GPA/metadata_classification.json'

        metadata_dict = load_json(metadata_pth)
        regime = metadata_dict['regime']
        sub_regime = metadata_dict['sub_regime']
        columns = metadata_dict['columns']
                    
        include_sensitive_columns = False
        include_intercept_term = False
        regime='supervised_learning'

        model_class = LogisticRegressionModel

        # Mean squared error
        primary_objective = model_class().default_objective

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

        return dataset,model_class,primary_objective,parse_trees
    
    return generate_dataset

@pytest.fixture
def RL_gridworld_dataset():
    from seldonian.RL.environments import gridworld3x3
    from seldonian.models.models import TabularSoftmaxModel

    def generate_dataset(constraint_strs,deltas):
        env = gridworld3x3.Environment()
        regime='reinforcement_learning'

        model_class = TabularSoftmaxModel
        model_instance = model_class(env)
        # Mean squared error
        primary_objective = model_instance.sample_IS_estimate

        # Load data into dataset
        data_pth = 'static/datasets/RL/gridworld/gridworld3x3_250episodes_list.pkl'
        metadata_pth = 'static/datasets/RL/gridworld/gridworld3x3_metadata.json'

        loader = DataSetLoader(
            regime=regime)

        dataset = loader.load_RL_dataset_from_episode_list(
            filename=data_pth,
            metadata_filename=metadata_pth)

        # For each constraint, make a parse tree
        parse_trees = []
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]

            delta = deltas[ii]
            # Create parse tree object
            parse_tree = ParseTree(delta=delta,
                regime='reinforcement_learning',sub_regime='all')

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