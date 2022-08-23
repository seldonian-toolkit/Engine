# createSpec.py
import os
from seldonian.parse_tree.parse_tree import make_parse_trees_from_constraints
from seldonian.dataset import DataSetLoader
from seldonian.utils.io_utils import load_json,save_pickle
from seldonian.spec import SupervisedSpec
from seldonian.models.models import LogisticRegressionModel

if __name__ == '__main__':
    data_pth = "../../static/datasets/supervised/GPA/gpa_classification_dataset.csv"
    metadata_pth = "../../static/datasets/supervised/GPA/metadata_classification.json"
    save_base_dir = '../../../interface_outputs'
    # Load metadata
    metadata_dict = load_json(metadata_pth)

    regime = metadata_dict['regime']
    columns = metadata_dict['columns']
    sensitive_columns = metadata_dict['sensitive_columns']
    sub_regime = metadata_dict['sub_regime']
    
    # Use logistic regression model
    model_class = LogisticRegressionModel
    
    # Set the primary objective to be log loss
    primary_objective = model_class().sample_logistic_loss

    # Load dataset from file
    loader = DataSetLoader(
        regime=regime)

    dataset = loader.load_supervised_dataset(
        filename=data_pth,
        metadata_filename=metadata_pth,
        file_type='csv')
    
    # Behavioral constraints
    deltas = [0.05]
    for constraint_name in ["disparate_impact",
        "demographic_parity","equalized_odds",
        "equal_opportunity","predictive_equality"]:
        save_dir = os.path.join(save_base_dir,f'gpa_{constraint_name}')
        os.makedirs(save_dir,exist_ok=True)
        # Define behavioral constraints
        if constraint_name == 'disparate_impact':
            constraint_strs = ['0.8 - min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M]))'] 
        elif constraint_name == 'demographic_parity':
            constraint_strs = ['abs((PR | [M]) - (PR | [F])) <= 0.2']
        elif constraint_name == 'equalized_odds':
            constraint_strs = ['abs((FNR | [M]) - (FNR | [F])) + abs((FPR | [M]) - (FPR | [F])) <= 0.35']
        elif constraint_name == 'equal_opportunity':
            constraint_strs = ['abs((FNR | [M]) - (FNR | [F])) <= 0.2']
        elif constraint_name == 'predictive_equality':
            constraint_strs = ['abs((FPR | [M]) - (FPR | [F])) <= 0.2']

        parse_trees = make_parse_trees_from_constraints(
            constraint_strs,
            deltas,
            regime='supervised_learning',
            sub_regime='classification',
            columns=columns)
        
        # Save spec object, using defaults where necessary
        spec = SupervisedSpec(
            dataset=dataset,
            model_class=model_class,
            parse_trees=parse_trees,
            sub_regime='classification',
            frac_data_in_safety=0.6,
            primary_objective=primary_objective,
            initial_solution_fn=model_class().fit,
            use_builtin_primary_gradient_fn=True,
            optimization_technique='gradient_descent',
            optimizer='adam',
            optimization_hyperparams={
                'lambda_init'   : 0.5,
                'alpha_theta'   : 0.01,
                'alpha_lamb'    : 0.01,
                'beta_velocity' : 0.9,
                'beta_rmsprop'  : 0.95,
                'num_iters'     : 1000,
                'gradient_library': "autograd",
                'hyper_search'  : None,
                'verbose'       : True,
            }
        )

        spec_save_name = os.path.join(save_dir,'spec.pkl')
        save_pickle(spec_save_name,spec,verbose=True)