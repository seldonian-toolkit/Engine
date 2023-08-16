# createSpec.py
import os
from seldonian.parse_tree.parse_tree import make_parse_trees_from_constraints
from seldonian.dataset import DataSetLoader
from seldonian.utils.io_utils import (load_json,save_pickle)
from seldonian.spec import createSupervisedSpec
from seldonian.models.models import (
    BinaryLogisticRegressionModel as LogisticRegressionModel)
from seldonian.models import objectives

if __name__ == '__main__':
    data_pth = "../../static/datasets/supervised/GPA/gpa_classification_dataset.csv"
    metadata_pth = "../../static/datasets/supervised/GPA/metadata_classification.json"
    save_base_dir = '../../../interface_outputs'
    # save_base_dir='.'
    # Load metadata
    regime='supervised_learning'
    sub_regime='classification'

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

        createSupervisedSpec(
            dataset=dataset,
            metadata_pth=metadata_pth,
            constraint_strs=constraint_strs,
            deltas=deltas,
            save_dir=save_dir,
            save=True,
            verbose=True)