import importlib
import autograd.numpy as np

from seldonian.utils.io_utils import load_json
from seldonian.parse_tree.parse_tree import ParseTree
from seldonian.dataset import DataSetLoader
from seldonian.models.model import (LinearRegressionModel,
    TabularSoftmaxModel)
from seldonian.spec import RLSpec, SupervisedSpec
from seldonian.seldonian_algorithm import seldonian_algorithm



### Begin tests

def test_gpa_data_regression():
    """ Test that the gpa regression example runs 
    with a simple non-conflicting constraint. Make
    sure safety test passes and solution is correct.
    """
    # Load metadata
    np.random.seed(0) 
    data_pth = 'static/datasets/GPA/gpa_regression_dataset.csv'
    metadata_pth = 'static/datasets/GPA/metadata_regression.json'

    metadata_dict = load_json(metadata_pth)
    regime = metadata_dict['regime']
    sub_regime = metadata_dict['sub_regime']
    columns = metadata_dict['columns']
    sensitive_columns = metadata_dict['sensitive_columns']
    label_column = metadata_dict['label_column']
                
    include_sensitive_columns = False
    include_intercept_term = True
    frac_data_in_safety = 0.6

    model_class = LinearRegressionModel

    # Mean squared error
    primary_objective = model_class().sample_Mean_Squared_Error

    # Load dataset from file
    loader = DataSetLoader(
        column_names=columns,
        sensitive_column_names=sensitive_columns,
        include_sensitive_columns=include_sensitive_columns,
        include_intercept_term=include_intercept_term,
        label_column=label_column,
        regime=regime)

    dataset = loader.from_csv(data_pth)
    
    constraint_strs = ['Mean_Squared_Error - 2.0'] 
    
    deltas = [0.05]

    # For each constraint, make a parse tree
    parse_trees = []
    for ii in range(len(constraint_strs)):
        constraint_str = constraint_strs[ii]

        delta = deltas[ii]
        # Create parse tree object
        parse_tree = ParseTree(delta=delta)

        # Fill out tree
        parse_tree.create_from_ast(constraint_str)
        # assign deltas for each base node
        # use equal weighting for each base node
        parse_tree.assign_deltas(weight_method='equal')

        # Assign bounds needed on the base nodes
        parse_tree.assign_bounds_needed()
        
        parse_trees.append(parse_tree)

    # Create spec object
    spec = SupervisedSpec(
        dataset=dataset,
        model_class=model_class,
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        parse_trees=parse_trees,
        initial_solution_fn=model_class().fit,
        bound_method='ttest',
        optimization_technique='gradient_descent',
        optimizer='adam',
        optimization_hyperparams={
            'lambda_init'   : 0.5,
            'alpha_theta'   : 0.005,
            'alpha_lamb'    : 0.005,
            'beta_velocity' : 0.9,
            'beta_rmsprop'  : 0.95,
            'num_iters'     : 200,
            'hyper_search'  : None,
            'verbose'       : True,
        }
    )

    # Run seldonian algorithm
    passed_safety,candidate_solution = seldonian_algorithm(spec)
    assert passed_safety == True
    array_to_compare = np.array(
        [4.17863795e-01,-2.38988670e-04,  5.62655484e-04,
        2.02541591e-04, 2.53897770e-04,  4.11365885e-05,
        1.81167007e-03,1.23389238e-03,-4.58006355e-04,
        1.51706564e-04])
    assert np.allclose(candidate_solution,array_to_compare)

def test_use_custom_primary_gradient():
    """ Test that the gpa regression example runs 
    when using a custom primary gradient function.
    It is the same as the built-in but passed as 
    a custom function. Make
    sure safety test passes and solution is correct.
    """
    # Load metadata
    np.random.seed(0) 
    data_pth = 'static/datasets/GPA/gpa_regression_dataset.csv'
    metadata_pth = 'static/datasets/GPA/metadata_regression.json'

    metadata_dict = load_json(metadata_pth)
    regime = metadata_dict['regime']
    sub_regime = metadata_dict['sub_regime']
    columns = metadata_dict['columns']
    sensitive_columns = metadata_dict['sensitive_columns']
    label_column = metadata_dict['label_column']
                
    include_sensitive_columns = False
    include_intercept_term = True
    frac_data_in_safety = 0.6

    model_class = LinearRegressionModel

    # Mean squared error
    primary_objective = model_class().sample_Mean_Squared_Error

    def gradient_MSE(model,theta,X,Y):
        n = len(X)
        prediction = model.predict(theta,X) # vector of values
        err = prediction-Y
        return 2/n*np.dot(err,X)

    # Load dataset from file
    loader = DataSetLoader(
        column_names=columns,
        sensitive_column_names=sensitive_columns,
        include_sensitive_columns=include_sensitive_columns,
        include_intercept_term=include_intercept_term,
        label_column=label_column,
        regime=regime)

    dataset = loader.from_csv(data_pth)
    
    constraint_strs = ['Mean_Squared_Error - 2.0'] 
    
    deltas = [0.05]

    # For each constraint, make a parse tree
    parse_trees = []
    for ii in range(len(constraint_strs)):
        constraint_str = constraint_strs[ii]

        delta = deltas[ii]
        # Create parse tree object
        parse_tree = ParseTree(delta=delta)

        # Fill out tree
        parse_tree.create_from_ast(constraint_str)
        # assign deltas for each base node
        # use equal weighting for each base node
        parse_tree.assign_deltas(weight_method='equal')

        # Assign bounds needed on the base nodes
        parse_tree.assign_bounds_needed()
        
        parse_trees.append(parse_tree)

    # Create spec object
    spec = SupervisedSpec(
        dataset=dataset,
        model_class=model_class,
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=False,
        custom_primary_gradient_fn=gradient_MSE,
        parse_trees=parse_trees,
        initial_solution_fn=model_class().fit,
        bound_method='ttest',
        optimization_technique='gradient_descent',
        optimizer='adam',
        optimization_hyperparams={
            'lambda_init'   : 0.5,
            'alpha_theta'   : 0.005,
            'alpha_lamb'    : 0.005,
            'beta_velocity' : 0.9,
            'beta_rmsprop'  : 0.95,
            'num_iters'     : 200,
            'hyper_search'  : None,
            'verbose'       : True,
        }
    )

    # Run seldonian algorithm
    passed_safety,candidate_solution = seldonian_algorithm(spec)
    assert passed_safety == True
    array_to_compare = np.array(
        [4.17863795e-01,-2.38988670e-04,  5.62655484e-04,
        2.02541591e-04, 2.53897770e-04,  4.11365885e-05,
        1.81167007e-03,1.23389238e-03,-4.58006355e-04,
        1.51706564e-04])
    assert np.allclose(candidate_solution,array_to_compare)

def test_RL_gridworld():
    """ Test that the RL gridworld example runs 
    with a simple performance improvement constraint. Make
    sure safety test passes and solution is correct.
    """
    # Load data and metadata
    np.random.seed(0) 
    data_pth = 'static/datasets/RL/gridworld/gridworld3x3_50episodes.csv'
    metadata_pth = 'static/datasets/RL/gridworld/gridworld3x3_metadata.json'

    metadata_dict = load_json(metadata_pth)
    regime = metadata_dict['regime']
    columns = metadata_dict['columns']
                
    include_sensitive_columns = False
    include_intercept_term = False
    frac_data_in_safety = 0.6

    # Model
    model_class = TabularSoftmaxModel

    # RL environment
    RL_environment_name = metadata_dict['RL_environment_name']
    RL_environment_module = importlib.import_module(
        f'seldonian.RL.environments.{RL_environment_name}')
    RL_environment_obj = RL_environment_module.Environment()    

    # Primary objective
    model_instance = model_class(RL_environment_obj)

    primary_objective = model_instance.default_objective

    # Load dataset from file
    loader = DataSetLoader(
        column_names=columns,
        sensitive_column_names=[],
        include_sensitive_columns=include_sensitive_columns,
        include_intercept_term=include_intercept_term,
        label_column='',
        regime=regime)

    dataset = loader.from_csv(data_pth)
    
    constraint_strs = ['-0.25 - J_pi_new'] 
    
    deltas = [0.05]

    # For each constraint, make a parse tree
    parse_trees = []
    for ii in range(len(constraint_strs)):
        constraint_str = constraint_strs[ii]

        delta = deltas[ii]
        # Create parse tree object
        parse_tree = ParseTree(delta=delta)

        # Fill out tree
        parse_tree.create_from_ast(constraint_str)
        # assign deltas for each base node
        # use equal weighting for each base node
        parse_tree.assign_deltas(weight_method='equal')

        # Assign bounds needed on the base nodes
        parse_tree.assign_bounds_needed()
        
        parse_trees.append(parse_tree)

    # # Create spec object
    spec = RLSpec(
        dataset=dataset,
        model_class=model_class,
        frac_data_in_safety=0.8,
        use_builtin_primary_gradient_fn=False,
        primary_objective=primary_objective,
        parse_trees=parse_trees,
        RL_environment_obj=RL_environment_obj,
        initial_solution_fn=None,
        bound_method='ttest',
        optimization_technique='gradient_descent',
        optimizer='adam',
        optimization_hyperparams={
            'lambda_init'   : 0.5,
            'alpha_theta'   : 0.005,
            'alpha_lamb'    : 0.005,
            'beta_velocity' : 0.9,
            'beta_rmsprop'  : 0.95,
            'num_iters'     : 20,
            'hyper_search'  : None,
            'verbose'       : True,
        },
        regularization_hyperparams={'reg_coef':0.1},
        normalize_returns=False,
    )

    # # Run seldonian algorithm
    passed_safety,candidate_solution = seldonian_algorithm(spec)
    assert passed_safety == True
    array_to_compare = np.array(
        [ 0.15982427, -0.16025631, -0.15884394,  0.16005764, -0.15922789,  0.15970923,
         -0.16008555,  0.15931073,  0.16045895, -0.16086362, -0.15997721,  0.15984026,
          0.15972855,  0.15871099,  0.16169017, -0.15966671,  0.15954723, -0.16046687,
         -0.15976241,  0.16024623,  0.15842995, -0.15832812, -0.16488758,  0.15972792,
         -0.16127811,  0.1607053,   0.16499328,  0.16099007, -0.16009257, -0.15898218,
          0.15874795,  0.1595269 ])
    assert np.allclose(candidate_solution,array_to_compare)
