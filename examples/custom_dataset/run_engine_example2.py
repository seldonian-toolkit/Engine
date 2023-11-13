# createSpec.py
import autograd.numpy as np
import os
import pandas as pd
from seldonian.parse_tree.parse_tree import ParseTree
from seldonian.parse_tree import zhat_funcs

from seldonian.dataset import CustomDataSet, load_custom_metadata, CustomMetaData
from seldonian.utils.io_utils import (load_json,save_pickle)
from seldonian.spec import Spec
from seldonian.models.models import (
    BinaryLogisticRegressionModel as LogisticRegressionModel) 
from seldonian.models import objectives
from seldonian.seldonian_algorithm import SeldonianAlgorithm
import seldonian.models.custom_text_model 

if __name__ == '__main__':
    # Load some string data in as lists of lists
    N = 100
    l=[chr(x) for x in np.random.randint(97,122,N)] # lowercase letters
    data = [l[i*3:i*3+3] for i in range(N//3)]

    all_col_names = ["string"]
    meta = CustomMetaData(all_col_names=all_col_names)
    dataset = CustomDataSet(data=data, sensitive_attrs=[], num_datapoints=len(data), meta=meta)
    # Create dataset from data and metadata file
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
   
    sensitive_col_names = []

    # Use custom text model
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
    constraint_str = 'CUST_LOSS <= 20.0'
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
        frac_data_in_safety=0.6,
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
            'num_iters'     : 400,
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