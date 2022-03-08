from src.parse_tree import *
from src.dataset import *
from src.safety_test import SafetyTest
from sklearn.model_selection import train_test_split
import pytest

### Begin tests

def test_safety_test(generate_data):
    # dummy data for linear regression
    np.random.seed(0)
    numPoints=1000
    X,Y = generate_data(numPoints,
        loc_X=0.0,loc_Y=0.0,sigma_X=1.0,sigma_Y=1.0)
    rows = np.hstack([np.expand_dims(X,axis=1),np.expand_dims(Y,axis=1)])
    df = pd.DataFrame(rows,columns=['feature1','label'])
    dataset = DataSet(df,meta_information=['feature1','label'],
        regime='supervised',label_column='label')
    candidate_df, safety_df = train_test_split(
            df, test_size=0.5, shuffle=False)

    safety_dataset = DataSet(
        safety_df,meta_information=['feature1','label'],
        regime='supervised',label_column='label')

    # Linear regression model
    from src.model import LRModel
    model_instance = LRModel()
    
    # One constraint, so one parse tree
    constraint_str1 = 'Mean_Squared_Error - 2.0'
    delta = 0.05 
    parse_trees = []
    pt = ParseTree(delta)
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


