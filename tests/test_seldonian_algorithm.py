import pytest
import importlib
import autograd.numpy as np

from seldonian.utils.io_utils import load_json
from seldonian.parse_tree.parse_tree import ParseTree
from seldonian.dataset import (DataSetLoader,
    SupervisedDataSet,RLDataSet)
from seldonian.models.model import (LinearRegressionModel,
    TabularSoftmaxModel)
from seldonian.spec import RLSpec, SupervisedSpec
from seldonian.seldonian_algorithm import seldonian_algorithm



### Begin tests

def test_gpa_data_regression(gpa_regression_dataset):
    """ Test that the gpa regression example runs 
    with a simple non-conflicting constraint. Make
    sure safety test passes and solution is correct.
    """
    rseed=0
    np.random.seed(rseed) 
    constraint_strs = ['Mean_Squared_Error - 2.0']
    deltas = [0.05]

    (dataset,model_class,
        primary_objective,parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs,
        deltas=deltas)

    frac_data_in_safety=0.6

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
        [ 4.17214537e-01, -1.59553688e-04,  6.32496667e-04,  2.61407098e-04,
	3.09777789e-04,  1.02968565e-04,  1.86971042e-03,  1.29303914e-03,
	-3.80396712e-04,  2.27989618e-04])

    assert np.allclose(candidate_solution,array_to_compare)

def test_gpa_data_regression_multiple_constraints(gpa_regression_dataset):
    """ Test that the gpa regression example runs 
    with a two constraints using gradient descent. Make
    sure safety test passes and solution is correct.
    """
    # Load metadata
    rseed=0
    np.random.seed(rseed) 
    constraint_strs = ['Mean_Squared_Error - 5.0','2.0 - Mean_Squared_Error']
    deltas = [0.05,0.1]

    (dataset,model_class,
        primary_objective,parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs,
        deltas=deltas)

    frac_data_in_safety=0.6

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
        [ 4.17453469e-01,  7.68395649e-05,  8.67557891e-04,  4.93930585e-04,
  5.42100872e-04,  3.37273577e-04,  2.10366433e-03,  1.52437236e-03,
 -1.44621631e-04,  4.65147476e-04]
    )
    assert np.allclose(candidate_solution,array_to_compare)

def test_NSF(gpa_regression_dataset):
    """ Test that no solution is found for a constraint
    that is impossible to satisfy, e.g. negative mean squared error 
    """
    rseed=0
    np.random.seed(rseed) 
    constraint_strs = ['Mean_Squared_Error + 2.0'] 
    deltas = [0.05]

    (dataset,model_class,
        primary_objective,parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs,
        deltas=deltas)

    frac_data_in_safety=0.6

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
        },
    )

    # Run seldonian algorithm
    passed_safety,candidate_solution = seldonian_algorithm(spec)
    assert passed_safety == False
    assert candidate_solution == 'NSF'

def test_black_box_optimizers(gpa_regression_dataset):
    """ Test that the black box optimizers successfully optimize the GPA 
    regression problem with a simple non-conflicting constraint
    """
    rseed=99
    np.random.seed(rseed) 
    constraint_strs = ['Mean_Squared_Error - 2.0']
    deltas = [0.05]
    (dataset,model_class,
        primary_objective,parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs,
        deltas=deltas)

    frac_data_in_safety=0.6

    array_to_compare = np.array([4.17214561e-01, -1.59553688e-04,  6.32496667e-04,  2.61407098e-04,
  3.09777789e-04,  1.02968565e-04,  1.86971038e-03,  1.29303914e-03,
 -3.80396712e-04,  2.27989618e-04])

    for optimizer in ['Powell','CG','Nelder-Mead','BFGS','CMA-ES']:
        spec = SupervisedSpec(
            dataset=dataset,
            model_class=model_class,
            frac_data_in_safety=frac_data_in_safety,
            primary_objective=primary_objective,
            use_builtin_primary_gradient_fn=False,
            parse_trees=parse_trees,
            initial_solution_fn=model_class().fit,
            bound_method='ttest',
            optimization_technique='barrier_function',
            optimizer=optimizer,
            optimization_hyperparams={
                'maxiter'   : 100 if optimizer == 'CMA-ES' else 1000,
                'seed':rseed,
                'hyper_search'  : None,
                'verbose'       : True,
            },
        )


        # Run seldonian algorithm
        passed_safety,candidate_solution = seldonian_algorithm(spec)

        assert passed_safety == True
        if optimizer != 'CMA-ES':
            # CMA-ES might come up with a different solution on test server
            assert np.allclose(candidate_solution,array_to_compare)

    # Test that a bad string for the optimizer raises an exception
    bad_optimizer = 'bad-optimizer'
    with pytest.raises(NotImplementedError) as excinfo:
        bad_spec = SupervisedSpec(
                dataset=dataset,
                model_class=model_class,
                frac_data_in_safety=frac_data_in_safety,
                primary_objective=primary_objective,
                use_builtin_primary_gradient_fn=False,
                parse_trees=parse_trees,
                initial_solution_fn=model_class().fit,
                bound_method='ttest',
                optimization_technique='barrier_function',
                optimizer=bad_optimizer,
                optimization_hyperparams={
                    'maxiter'   : 1000,
                    'seed':rseed,
                    'hyper_search'  : None,
                    'verbose'       : True,
                },
            )

    # Run seldonian algorithm
    error_str = "Optimizer: bad-optimizer is not an acceptable optimizer"
    assert error_str in str(excinfo.value)

def test_use_custom_primary_gradient(gpa_regression_dataset):
    """ Test that the gpa regression example runs 
    when using a custom primary gradient function.
    It is the same as the built-in but passed as 
    a custom function. Make
    sure safety test passes and solution is correct.
    """

    def gradient_MSE(model,theta,X,Y):
        n = len(X)
        prediction = model.predict(theta,X) # vector of values
        err = prediction-Y
        return 2/n*np.dot(err,X)

    rseed=0
    np.random.seed(rseed) 
    constraint_strs = ['Mean_Squared_Error - 2.0'] 
    deltas = [0.05]

    (dataset,model_class,
        primary_objective,parse_trees) = gpa_regression_dataset(
        constraint_strs=constraint_strs,
        deltas=deltas)

    frac_data_in_safety=0.6

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
        [ 4.17214537e-01, -1.59553688e-04,  6.32496667e-04,  2.61407098e-04,
  3.09777789e-04,  1.02968565e-04,  1.86971042e-03,  1.29303914e-03,
 -3.80396712e-04,  2.27989618e-04])
    assert np.allclose(candidate_solution,array_to_compare)

# def test_RL_gridworld():
#     """ Test that the RL gridworld example runs 
#     with a simple performance improvement constraint. Make
#     sure safety test passes and solution is correct.
#     """
#     # Load data and metadata
#     rseed=99
#     np.random.seed(rseed) 
#     data_pth = 'static/datasets/RL/gridworld/gridworld3x3_500episodes.csv'
#     metadata_pth = 'static/datasets/RL/gridworld/gridworld3x3_metadata.json'

#     metadata_dict = load_json(metadata_pth)
#     regime = metadata_dict['regime']
#     columns = metadata_dict['columns']
                
#     include_sensitive_columns = False
#     include_intercept_term = False
#     frac_data_in_safety = 0.6

#     # Model
#     model_class = TabularSoftmaxModel

#     # RL environment
#     RL_environment_name = metadata_dict['RL_environment_name']
#     RL_environment_module = importlib.import_module(
#         f'seldonian.RL.environments.{RL_environment_name}')
#     RL_environment_obj = RL_environment_module.Environment()    

#     # Primary objective
#     model_instance = model_class(RL_environment_obj)

#     primary_objective = model_instance.default_objective

#     # Load dataset from file
#     loader = DataSetLoader(
#         column_names=columns,
#         sensitive_column_names=[],
#         include_sensitive_columns=include_sensitive_columns,
#         include_intercept_term=include_intercept_term,
#         label_column='',
#         regime=regime)

#     dataset = loader.from_csv(data_pth)
    
#     constraint_strs = ['-0.25 - J_pi_new'] 
    
#     deltas = [0.05]

#     # For each constraint, make a parse tree
#     parse_trees = []
#     for ii in range(len(constraint_strs)):
#         constraint_str = constraint_strs[ii]

#         delta = deltas[ii]
#         # Create parse tree object
#         parse_tree = ParseTree(delta=delta,regime='RL',
#         sub_regime='all')

#         # Fill out tree
#         parse_tree.create_from_ast(constraint_str)
#         # assign deltas for each base node
#         # use equal weighting for each base node
#         parse_tree.assign_deltas(weight_method='equal')

#         # Assign bounds needed on the base nodes
#         parse_tree.assign_bounds_needed()
        
#         parse_trees.append(parse_tree)

#     # # Create spec object
#     spec = RLSpec(
#         dataset=dataset,
#         model_class=model_class,
#         frac_data_in_safety=frac_data_in_safety,
#         use_builtin_primary_gradient_fn=False,
#         primary_objective=primary_objective,
#         parse_trees=parse_trees,
#         RL_environment_obj=RL_environment_obj,
#         initial_solution_fn=None,
#         bound_method='ttest',
#         optimization_technique='gradient_descent',
#         optimizer='adam',
#         optimization_hyperparams={
#             'alpha_theta'   : 0.005,
#             'alpha_lamb'    : 0.005,
#             'beta_velocity' : 0.9,
#             'beta_rmsprop'  : 0.95,
#             'num_iters'     : 10,
#             'hyper_search'  : None,
#             'verbose'       : True,
#         },
#         normalize_returns=False,
#     )

#     # # Run seldonian algorithm
#     passed_safety,candidate_solution = seldonian_algorithm(spec)
#     assert passed_safety == True
#     array_to_compare = np.array(
#        [ 0.10071358, -0.10060849,  0.10067788,  0.10049752,  0.1007003,  -0.10076564,
#         -0.10071716,  0.10075474,  0.10062417, -0.10072251, -0.10084294,  0.10104625,
#         0.10069145, -0.10065914,  0.10056431, -0.10045578,  0.10069424, -0.10063767,
#         -0.10576197,  0.1006481,   0.09996722,  0.10104243, -0.10079048,  0.10456359,
#         0.10064326,  0.10054124,  0.10313019, -0.10062128, -0.1008367,  -0.10074637,
#         -0.10040743,  0.10061044])
#     assert np.allclose(candidate_solution,array_to_compare)

def test_RL_builtin_or_custom_gradient_not_supported():
    """ Test that an error is raised if user tries to 
    use built-in gradient or a custom gradient 
    when doing RL
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
        regime=regime)

    dataset = loader.load_RL_dataset(
        filename=data_pth,
        metadata_filename=metadata_pth,
        file_type='csv')
    
    constraint_strs = ['-0.25 - J_pi_new'] 
    
    deltas = [0.05]

    # For each constraint, make a parse tree
    parse_trees = []
    for ii in range(len(constraint_strs)):
        constraint_str = constraint_strs[ii]

        delta = deltas[ii]
        # Create parse tree object
        parse_tree = ParseTree(delta=delta,regime='RL',
        sub_regime='all')

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
        use_builtin_primary_gradient_fn=True,
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

    # Run seldonian algorithm, making sure we capture error
    error_str = ("Using a builtin primary objective gradient"
                " is not yet supported for regimes other"
                " than supervised learning")
    with pytest.raises(NotImplementedError) as excinfo:
        passed_safety,candidate_solution = seldonian_algorithm(spec)
        
    assert error_str in str(excinfo.value)

    # # Create spec object
    spec2 = RLSpec(
        dataset=dataset,
        model_class=model_class,
        frac_data_in_safety=0.8,
        use_builtin_primary_gradient_fn=False,
        custom_primary_gradient_fn=lambda x: x,
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

    # Run seldonian algorithm, making sure we capture error
    error_str2 = ("Using a provided primary objective gradient"
                " is not yet supported for regimes other"
                " than supervised learning")
    with pytest.raises(NotImplementedError) as excinfo2:
        passed_safety,candidate_solution = seldonian_algorithm(spec2)
        
    assert error_str2 in str(excinfo2.value)
    

