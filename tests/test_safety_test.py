from seldonian.parse_tree.parse_tree import *
from seldonian.dataset import *
from seldonian.safety_test.safety_test import SafetyTest
from sklearn.model_selection import train_test_split
import pytest

### Begin tests

def test_run_safety_test(generate_data):
    # dummy data for linear regression
    np.random.seed(0)
    numPoints=1000
    X,Y = generate_data(numPoints,
        loc_X=0.0,loc_Y=0.0,sigma_X=1.0,sigma_Y=1.0)
    rows = np.hstack([np.expand_dims(X,axis=1),np.expand_dims(Y,axis=1)])
    df = pd.DataFrame(rows,columns=['feature1','label'])
    columns = ['feature1','label']
    label_column = 'label'
    regime = 'supervised'
    include_sensitive_columns=False
    include_intercept_term=True
    dataset = SupervisedDataSet(df,meta_information=columns,
        label_column='label',
        include_sensitive_columns=include_sensitive_columns,
        include_intercept_term=include_intercept_term)

    candidate_df, safety_df = train_test_split(
            df, test_size=0.5, shuffle=False)

    safety_dataset = SupervisedDataSet(
        safety_df,meta_information=columns,
        regime=regime,label_column='label',
        include_sensitive_columns=include_sensitive_columns,
        include_intercept_term=include_intercept_term)

    # Linear regression model
    from seldonian.models.models import LinearRegressionModel
    model_instance = LinearRegressionModel()
    
    # One constraint, so one parse tree
    constraint_str1 = 'Mean_Squared_Error - 2.0'
    delta = 0.05 
    parse_trees = []
    pt = ParseTree(delta,regime='supervised',
        sub_regime='regression')
    pt.create_from_ast(constraint_str1)
    pt.assign_deltas(weight_method='equal')
    parse_trees.append(pt)

    # A candidate solution that we know should fail
    candidate_solution = np.array([20,4])

    st = SafetyTest(safety_dataset,model_instance,parse_trees)
    passed_safety = st.run(candidate_solution,bound_method='ttest')
    assert passed_safety == False
    
    # A candidate solution that we know should pass,
    candidate_solution = np.array([0,1])
    passed_safety = st.run(candidate_solution,bound_method='ttest')
    assert passed_safety == True

def test_evaluate_primary_objective(generate_data):
    """ Test evaluating the primary objective 
    using solutions on the safety dataset """ 
    np.random.seed(0)
    numPoints=1000
    X,Y = generate_data(numPoints,
        loc_X=0.0,loc_Y=0.0,sigma_X=1.0,sigma_Y=1.0)
    rows = np.hstack([np.expand_dims(X,axis=1),np.expand_dims(Y,axis=1)])
    df = pd.DataFrame(rows,columns=['feature1','label'])
    columns = ['feature1','label']
    label_column = 'label'
    regime = 'supervised'
    include_sensitive_columns=False
    include_intercept_term=True
    dataset = SupervisedDataSet(df,meta_information=columns,
        label_column='label',
        include_sensitive_columns=include_sensitive_columns,
        include_intercept_term=include_intercept_term)

    candidate_df, safety_df = train_test_split(
            df, test_size=0.5, shuffle=False)

    safety_dataset = SupervisedDataSet(
        safety_df,meta_information=columns,
        regime=regime,label_column='label',
        include_sensitive_columns=include_sensitive_columns,
        include_intercept_term=include_intercept_term)

    # Linear regression model
    from seldonian.models.models import LinearRegressionModel
    model_instance = LinearRegressionModel()
    primary_objective = model_instance.default_objective
    
    # One constraint, so one parse tree
    constraint_str1 = 'Mean_Squared_Error - 2.0'
    delta = 0.05 
    parse_trees = []
    pt = ParseTree(delta,regime='supervised',
        sub_regime='regression')
    pt.create_from_ast(constraint_str1)
    pt.assign_deltas(weight_method='equal')
    parse_trees.append(pt)

    # A candidate solution that we know is bad
    solution = np.array([20,4])

    st = SafetyTest(safety_dataset,model_instance,parse_trees)
    po_eval = st.evaluate_primary_objective(solution,primary_objective)
    assert po_eval == pytest.approx(401.899175)
    
    # A candidate solution that we know is good
    solution = np.array([0,1])
    po_eval = st.evaluate_primary_objective(solution,primary_objective)
    assert po_eval == pytest.approx(0.94263425)


