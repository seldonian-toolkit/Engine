import os,sys
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from seldonian.parse_tree.parse_tree import *
from seldonian.dataset import *
from seldonian.utils.io_utils import load_json
from seldonian.seldonian_algorithm import seldonian_algorithm
from seldonian.models.model import LinearRegressionModel
from seldonian.spec import SupervisedSpec

def gradient_MSE(model,theta,X,Y):
    n = len(X)
    prediction = model.predict(theta,X) # vector of values
    err = prediction-Y
    return 2/n*np.dot(err,X)

if __name__ == '__main__':
	# gpa dataset
	# Load metadata 
    data_pth = 'static/datasets/supervised/GPA/gpa_regression_dataset.csv'
    metadata_pth = 'static/datasets/supervised/GPA/metadata_regression.json'

    metadata_dict = load_json(metadata_pth)
    regime = metadata_dict['regime']
    sub_regime = metadata_dict['sub_regime']
    columns = metadata_dict['columns']
    sensitive_columns = metadata_dict['sensitive_columns']
                
    include_sensitive_columns = False
    include_intercept_term = True
    frac_data_in_safety = 0.6

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
    
    # constraint_strs = ['Mean_Squared_Error - 5.0','2.0 - Mean_Squared_Error'] 
    constraint_strs = ['abs((Mean_Error|[M]) - (Mean_Error|[F])) - 0.1']
    
    deltas = [0.05]

    # For each constraint, make a parse tree
    parse_trees = []
    for ii in range(len(constraint_strs)):
        constraint_str = constraint_strs[ii]

        delta = deltas[ii]
        # Create parse tree object
        parse_tree = ParseTree(delta=delta,regime='supervised',
        sub_regime='regression',columns=sensitive_columns)

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
        # custom_primary_gradient_fn=gradient_MSE,
        parse_trees=parse_trees,
        initial_solution_fn=model_class().fit,
        bound_method='ttest',
        optimization_technique='gradient_descent',
        optimizer='adam',
        optimization_hyperparams={
            'lambda_init'   : np.array([0.5]),
            'alpha_theta'   : 0.01,
            'alpha_lamb'    : 0.01,
            'beta_velocity' : 0.9,
            'beta_rmsprop'  : 0.95,
            'num_iters'     : 600,
            'gradient_library': "autograd",
            'hyper_search'  : None,
            'verbose'       : True,
        }
    )

    # Run seldonian algorithm
    passed_safety,candidate_solution = seldonian_algorithm(spec)
    print(passed_safety,candidate_solution)
