# createSpec.py
import autograd.numpy as np
import os
import pandas as pd
from seldonian.parse_tree.parse_tree import make_parse_trees_from_constraints

from seldonian.dataset import SupervisedDataSet, DataSetLoader
from seldonian.utils.io_utils import (load_json,save_pickle)
from seldonian.spec import SupervisedSpec
from seldonian.models.models import (
    BinaryLogisticRegressionModel as LogisticRegressionModel) 
from seldonian.models import objectives
from seldonian.seldonian_algorithm import SeldonianAlgorithm

from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    data_pth = "../../static/datasets/supervised/german_credit/german_loan_numeric_forseldonian.csv"
    metadata_pth = "../../static/datasets/supervised/german_credit/metadata_german_loan.json"
    save_dir = '.'
    os.makedirs(save_dir,exist_ok=True)
    # Create dataset from data and metadata file
    regime='supervised_learning'
    sub_regime='classification'

    loader = DataSetLoader(
        regime=regime)

    orig_dataset = loader.load_supervised_dataset(
        filename=data_pth,
        metadata_filename=metadata_pth,
        file_type='csv')
    # Split into candidate and safety here
    orig_features = orig_dataset.features
    orig_labels = orig_dataset.labels
    orig_sensitive_attrs = orig_dataset.sensitive_attrs

    (candidate_features,safety_features,candidate_labels,
    safety_labels,candidate_sensitive_attrs,
    safety_sensitive_attrs
        ) = train_test_split(
            orig_features,
            orig_labels,
            orig_sensitive_attrs,
            shuffle=True,
            test_size=0.6,
            random_state=42)
    
    candidate_dataset = SupervisedDataSet(
        features=candidate_features, 
        labels=candidate_labels,
        sensitive_attrs=candidate_sensitive_attrs, 
        num_datapoints=len(candidate_features),
        meta=orig_dataset.meta)

    safety_dataset = SupervisedDataSet(
        features=safety_features, 
        labels=safety_labels,
        sensitive_attrs=safety_sensitive_attrs, 
        num_datapoints=len(safety_features),
        meta=orig_dataset.meta)

    sensitive_col_names = orig_dataset.meta.sensitive_col_names

    def func(model,X,Y):
        print("X:")
        print(X)
        return model.fit(X,Y)
    # Use logistic regression model
    model = LogisticRegressionModel()
    
    # Set the primary objective to be log loss
    primary_objective = objectives.binary_logistic_loss
    
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
        dataset=None,
        candidate_dataset=candidate_dataset,
        safety_dataset=safety_dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime=sub_regime,
        frac_data_in_safety=None,
        primary_objective=primary_objective,
        initial_solution_fn=func,
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
   
   
    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run(debug=True,write_cs_logfile=True)


    # spec_save_name = os.path.join(save_dir,f'loans_{constraint_name}_{epsilon}_spec.pkl')
    # save_pickle(spec_save_name,spec)
    # print(f"Saved Spec object to: {spec_save_name}")