from seldonian.parse_tree.parse_tree import *
from seldonian.dataset import *
from seldonian.safety_test.safety_test import SafetyTest
from seldonian.models import objectives
from seldonian.utils.tutorial_utils import generate_data

from sklearn.model_selection import train_test_split
import pytest

### Begin tests

def test_run_safety_test(simulated_regression_dataset):
    
    # One constraint, so one parse tree
    constraint_strs = ['Mean_Squared_Error - 2.0']
    deltas = [0.05]
    frac_data_in_safety=0.6

    (dataset,model,primary_objective,
        parse_trees) = simulated_regression_dataset(
            constraint_strs,deltas)

    features = dataset.features
    labels = dataset.labels

    (candidate_features, safety_features,
        candidate_labels, safety_labels) = train_test_split(
            features, labels,test_size=frac_data_in_safety, shuffle=False)
    
    safety_dataset = SupervisedDataSet(
        features=safety_features,
        labels=safety_labels,
        sensitive_attrs=[],
        num_datapoints=len(safety_features),
        meta_information=dataset.meta_information)

    # A candidate solution that we know should fail
    candidate_solution = np.array([20,4])

    st = SafetyTest(safety_dataset,model,parse_trees)
    passed_safety = st.run(candidate_solution)
    assert passed_safety == False
    
    # A candidate solution that we know should pass,
    candidate_solution = np.array([0,1])
    passed_safety = st.run(candidate_solution)
    assert passed_safety == True

def test_evaluate_primary_objective(simulated_regression_dataset):
    """ Test evaluating the primary objective 
    using solutions on the safety dataset """ 
    
    # One constraint, so one parse tree
    constraint_strs = ['Mean_Squared_Error - 2.0']
    deltas = [0.05]
    frac_data_in_safety=0.5

    (dataset,model,primary_objective,
        parse_trees) = simulated_regression_dataset(
            constraint_strs,deltas)

    features = dataset.features
    labels = dataset.labels

    (candidate_features, safety_features,
        candidate_labels, safety_labels) = train_test_split(
            features, labels,test_size=frac_data_in_safety, shuffle=False)
    
    safety_dataset = SupervisedDataSet(
        features=safety_features,
        labels=safety_labels,
        sensitive_attrs=[],
        num_datapoints=len(safety_features),
        meta_information=dataset.meta_information)

    # A candidate solution that we know is bad
    solution = np.array([20,4])
    st = SafetyTest(safety_dataset,model,parse_trees)
    primary_obj_evl = st.evaluate_primary_objective(solution,primary_objective)
    assert primary_obj_evl == pytest.approx(401.899175)
    
    # A candidate solution that we know is good
    solution = np.array([0,1])
    primary_obj_evl = st.evaluate_primary_objective(solution,primary_objective)
    assert primary_obj_evl == pytest.approx(0.94263425)


