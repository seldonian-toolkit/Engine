from seldonian.parse_tree.parse_tree import *
from seldonian.dataset import *
from seldonian.safety_test.safety_test import SafetyTest
from seldonian.models import objectives
from seldonian.RL.RL_model import RL_model
from seldonian.utils.tutorial_utils import generate_data

from sklearn.model_selection import train_test_split
import pytest

### Begin tests

def test_run_safety_test_regression(
    simulated_regression_dataset):
    
    # One constraint, so one parse tree
    constraint_strs = ['Mean_Squared_Error - 2.0']
    deltas = [0.05]
    frac_data_in_safety=0.6
    numPoints=1000
    (dataset,model,primary_objective,
        parse_trees) = simulated_regression_dataset(
            constraint_strs,deltas,numPoints=numPoints)

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
    # First without batching
    passed_safety = st.run(candidate_solution)
    assert passed_safety == True
    upper_bound_no_batching = parse_trees[0].root.upper
    assert upper_bound_no_batching == pytest.approx(
        -0.947692744102955)
    # Now with batching
    passed_safety_batching = st.run(candidate_solution,
        batch_size_safety=int(round(numPoints/10)))
    assert passed_safety_batching == True
    upper_bound_batching = parse_trees[0].root.upper
    assert upper_bound_batching == pytest.approx(
        -0.947692744102955)

def test_evaluate_primary_objective_regression(
    simulated_regression_dataset):
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

def test_evaluate_primary_objective_RL(
    RL_gridworld_dataset):
    """ Test evaluating the primary objective 
    using solutions on the safety dataset """ 
    
    # One constraint, so one parse tree
    rseed=99
    np.random.seed(rseed)
    regime='reinforcement_learning'
    constraint_strs = ['-0.25 - J_pi_new']
    deltas = [0.05]
    
    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,
        deltas,
        regime=regime,
        sub_regime='all',
        columns=[],
        delta_weight_method='equal')
    (dataset,policy,
        env_kwargs,primary_objective) = RL_gridworld_dataset()
                
    frac_data_in_safety = 0.6

    # Model

    model = RL_model(policy=policy,env_kwargs=env_kwargs)

    candidate_episodes, safety_episodes = train_test_split(
            dataset.episodes,
            test_size=frac_data_in_safety, shuffle=False)
    
    safety_dataset = RLDataSet(
        episodes=safety_episodes,
        num_datapoints=len(safety_episodes),
        meta_information=dataset.meta_information)

    # A candidate solution that we know is bad
    solution = np.zeros((9,4))
    st = SafetyTest(safety_dataset,model,parse_trees,
        regime=regime)
    primary_obj_evl = st.evaluate_primary_objective(
        solution,primary_objective)
    assert primary_obj_evl == pytest.approx(0.45250228)
    






