# createSpec.py
from seldonian.parse_tree.parse_tree import *
from seldonian.dataset import DataSetLoader
from seldonian.utils.io_utils import save_pickle
from seldonian.models.models import BinaryLogisticRegressionModel 
from seldonian.models import objectives
from seldonian.spec import SupervisedSpec

def initial_solution_fn(m,X,y):
    return m.fit(X,y)

if __name__ == '__main__':

    primary_cand_data_pth = "gpa_classification_primary_cand_dataset.csv"
    primary_safety_data_pth = "gpa_classification_primary_safety_dataset.csv"

    primary_metadata_pth = "primary_metadata_classification.json"

    addl_cand_data_pth = "gpa_classification_addl_cand_dataset.csv"
    addl_safety_data_pth = "gpa_classification_addl_safety_dataset.csv"
    addl_metadata_pth = "addl_metadata_classification.json"

    regime = "supervised_learning"
    sub_regime = "classification"
    # Load datasets from file
    loader = DataSetLoader(regime=regime)

    primary_cand_dataset = loader.load_supervised_dataset(
        filename=primary_cand_data_pth, 
        metadata_filename=primary_metadata_pth, 
        file_type="csv"
    )

    primary_safety_dataset = loader.load_supervised_dataset(
        filename=primary_safety_data_pth, 
        metadata_filename=primary_metadata_pth, 
        file_type="csv"
    )

    addl_cand_dataset = loader.load_supervised_dataset(
        filename=addl_cand_data_pth, 
        metadata_filename=addl_metadata_pth, 
        file_type="csv"
    )

    addl_safety_dataset = loader.load_supervised_dataset(
        filename=addl_safety_data_pth, 
        metadata_filename=addl_metadata_pth, 
        file_type="csv"
    )


    # Model, primary objective
    model = BinaryLogisticRegressionModel()
    primary_objective = objectives.binary_logistic_loss

    # Behavioral constraints
    constraint_strs = ['abs((PR | [M]) - (PR | [F])) - 0.2']
    deltas = [0.05] 
    # For each constraint, make a parse tree 
    parse_trees = []
    for ii in range(len(constraint_strs)):
        constraint_str = constraint_strs[ii]
        delta = deltas[ii]
        # Create parse tree object
        parse_tree = ParseTree(
            delta=delta,
            regime="supervised_learning",
            sub_regime=addl_cand_dataset.meta.sub_regime,
            columns=addl_cand_dataset.sensitive_col_names,
        )

        parse_tree.build_tree(constraint_str=constraint_str)
        parse_trees.append(parse_tree)


    additional_datasets = {}
    for pt in parse_trees:
        additional_datasets[pt.constraint_str] = {}
        base_nodes_this_tree = list(pt.base_node_dict.keys())
        for bn in base_nodes_this_tree:
            additional_datasets[pt.constraint_str][bn] = {
                "candidate_dataset": addl_cand_dataset,
                "safety_dataset": addl_safety_dataset
            }


    # Save spec object, using defaults where necessary
    spec = SupervisedSpec(
        dataset=None,
        candidate_dataset=primary_cand_dataset,
        safety_dataset=primary_safety_dataset,
        additional_datasets=additional_datasets,
        model=model,
        parse_trees=parse_trees,
        sub_regime=sub_regime,
        frac_data_in_safety=0.6,
        primary_objective=primary_objective,
        initial_solution_fn=initial_solution_fn,
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
            'num_iters'     : 1000,
            'gradient_library': "autograd",
            'hyper_search'  : None,
            'verbose'       : True,
        }
    )
    savename = "demographic_parity_addl_datasets_nodups.pkl"
    save_pickle(savename,spec,verbose=True)
            