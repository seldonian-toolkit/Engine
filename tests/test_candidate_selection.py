from seldonian.parse_tree.parse_tree import *
from seldonian.dataset import *
from seldonian.candidate_selection import CandidateSelection
from sklearn.model_selection import train_test_split
from seldonian.models.model import LinearRegressionModel
import pytest

### Begin tests

def test_simulated_data(generate_data):
    # dummy data for linear regression
    np.random.seed(0)
    numPoints=1000
    X,Y = generate_data(numPoints,
        loc_X=0.0,loc_Y=0.0,sigma_X=1.0,sigma_Y=1.0)
    rows = np.hstack([np.expand_dims(X,axis=1),np.expand_dims(Y,axis=1)])
    df = pd.DataFrame(rows,columns=['feature1','label'])
    candidate_df, safety_df = train_test_split(
            df, test_size=0.5, shuffle=False)
    label_column = 'label'
    include_sensitive_columns=False
    include_intercept_term=True,
    candidate_dataset = DataSet(
        candidate_df,meta_information=['feature1','label'],
        regime='supervised',label_column=label_column,
        include_sensitive_columns=include_sensitive_columns,
        include_intercept_term=include_intercept_term)
    candidate_labels = candidate_dataset.df[label_column]
    candidate_features = candidate_dataset.df.loc[:,
        candidate_dataset.df.columns != label_column]
    if not include_sensitive_columns:
        candidate_features = candidate_features.drop(
            columns=candidate_dataset.sensitive_column_names)

    if include_intercept_term:
        candidate_features.insert(0,'offset',1.0) # inserts a column of 1's

    safety_dataset = DataSet(
        safety_df,meta_information=['feature1','label'],
        regime='supervised',label_column='label',
        include_sensitive_columns=include_sensitive_columns,
        include_intercept_term=include_intercept_term)
    n_safety = len(safety_df)
    # Linear regression model
    model_instance = LinearRegressionModel()
    
    # One constraint, so one parse tree
    constraint_str1 = 'Mean_Squared_Error - 2.0'
    delta = 0.05
    parse_trees = []
    pt = ParseTree(delta)
    pt.create_from_ast(constraint_str1)
    pt.assign_deltas(weight_method='equal')
    parse_trees.append(pt)

    minimizer_options = {}
    initial_solution = model_instance.fit(
            candidate_features,candidate_labels)
    print(initial_solution)
    cs = CandidateSelection(
        model=model_instance,
        candidate_dataset=candidate_dataset,
        n_safety=n_safety,
        parse_trees=parse_trees,
        primary_objective=model_instance.sample_Mean_Squared_Error,
        optimization_technique='barrier_function',
        optimizer='Powell',
        initial_solution=initial_solution,
        minimizer_options=minimizer_options)
    candidate_solution = cs.run(minimizer_options=minimizer_options)
    assert candidate_solution[0] == pytest.approx(0.0280619)
    assert candidate_solution[1] == pytest.approx(0.95523116)


