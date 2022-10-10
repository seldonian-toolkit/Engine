# createSpec.py
import os
from seldonian.parse_tree.parse_tree import make_parse_trees_from_constraints
from seldonian.dataset import DataSetLoader
from seldonian.utils.io_utils import (load_json,
    load_supervised_metadata,save_pickle)
from seldonian.spec import createSupervisedSpec
from seldonian.models.models import LogisticRegressionModel
from seldonian.models import objectives

if __name__ == '__main__':
    data_pth = "../../static/datasets/supervised/GPA/gpa_multiclass_dataset.csv"
    metadata_pth = "../../static/datasets/supervised/GPA/metadata_classification.json"
    save_dir = '.'
    # Load metadata
    (regime, sub_regime, columns,
        sensitive_columns) = load_supervised_metadata(metadata_pth)

    # Load dataset from file
    loader = DataSetLoader(
        regime=regime)
    dataset = loader.load_supervised_dataset(
        filename=data_pth,
        metadata_filename=metadata_pth,
        file_type='csv',
        include_intercept_term=False)
    
    # Behavioral constraints
    deltas = [0.05]
            
    constraint_strs = ['abs((CM_[0,0] | [M]) - (CM_[0,0] | [F])) <= 0.1'] 

    spec = createSupervisedSpec(
        dataset=dataset,
        metadata_pth=metadata_pth,
        constraint_strs=constraint_strs,
        deltas=deltas,
        save_dir=save_dir,
        save=True,
        verbose=True)