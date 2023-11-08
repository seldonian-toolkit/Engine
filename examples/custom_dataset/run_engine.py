# createSpec.py
import autograd.numpy as np
import os
import pandas as pd
from seldonian.parse_tree.parse_tree import ParseTree
from seldonian.parse_tree import zhat_funcs

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
    model = LogisticRegressionModel()
    
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

    # Define behavioral constraints
    epsilon = 0.9
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
        print("in custom_vector_Positive_Rate()")
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


    # Use vanilla Spec object for custom datasets.
    spec = Spec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        frac_data_in_safety=0.6,
        primary_objective=custom_log_loss,
        initial_solution_fn=model.fit,
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
            'num_iters'     : 1500,
            'gradient_library': "autograd",
            'hyper_search'  : None,
            'verbose'       : True,
        }
    )
    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run(debug=True,write_cs_logfile=True)


    # spec_save_name = os.path.join(save_dir,f'loans_{constraint_name}_{epsilon}_spec.pkl')
    # save_pickle(spec_save_name,spec)
    # print(f"Saved Spec object to: {spec_save_name}")