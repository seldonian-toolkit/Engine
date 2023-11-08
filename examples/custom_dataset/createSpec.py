# createSpec.py
import autograd.numpy as np
import os
import pandas as pd
from seldonian.parse_tree.parse_tree import (ParseTree,
    make_parse_trees_from_constraints)

from seldonian.dataset import CustomDataSet, load_custom_metadata
from seldonian.utils.io_utils import (load_json,save_pickle)
from seldonian.spec import SupervisedSpec
from seldonian.models.models import (
    BinaryLogisticRegressionModel as LogisticRegressionModel) 
from seldonian.models import objectives



if __name__ == '__main__':
    data_pth = "../../static/datasets/custom/german_credit/german_loan_numeric_forseldonian.csv"
    metadata_pth = "../../static/datasets/custom/german_credit/metadata_german_loan.json"
    save_dir = '.'
    os.makedirs(save_dir,exist_ok=True)
    # Create dataset from data and metadata file
    regime='custom'

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

    # In this specific case we know that we have features and labels, 
    # and we are going to need them for evaluating the objective function 
    # and measure functions, 
    # but in general this may not be true. 
    feature_col_names = 
   
    sensitive_col_names = dataset.meta.sensitive_col_names

    # Use logistic regression model
    model = LogisticRegressionModel()
    
    # Define the primary objective to be log loss
    # Can just call the existing log loss function
    # but must wrap it because in this custom 
    # setting we don't know what features and labels
    # are a priori

    def custom_log_loss(model,theta,data,**kwargs):
        """Calculate average logistic loss
        over all data points for binary classification

        :param model: SeldonianModel instance
        :param theta: The parameter weights
        :type theta: numpy ndarray
        :param data: A list of samples, where samples could be anything

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
    epsilon = 0.9
    constraint_name = "disparate_impact"
    if constraint_name == "disparate_impact":
        constraint_strs = [f'min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M])) >= {epsilon}'] 
    deltas = [0.05]
    
    # For each constraint (in this case only one), make a parse tree
    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,deltas,regime=regime,
        sub_regime=sub_regime,columns=sensitive_col_names)

    # Save spec object, using defaults where necessary
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime=sub_regime,
        frac_data_in_safety=0.6,
        primary_objective=primary_objective,
        initial_solution_fn=model.fit,
        use_builtin_primary_gradient_fn=True,
        optimization_technique='gradient_descent',
        optimizer='adam',
        optimization_hyperparams={
            'lambda_init'   : np.array([0.5]),
            'alpha_theta'   : 0.01,
            'alpha_lamb'    : 0.01,
            'beta_velocity' : 0.9,
            'beta_rmsprop'  : 0.95,
            'use_batches'   : False,
            'num_iters'     : 1500,
            'gradient_library': "autograd",
            'hyper_search'  : None,
            'verbose'       : True,
        }
    )

    spec_save_name = os.path.join(save_dir,f'loans_{constraint_name}_{epsilon}_spec.pkl')
    save_pickle(spec_save_name,spec)
    print(f"Saved Spec object to: {spec_save_name}")