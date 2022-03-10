from src.parse_tree import *
from src.dataset import *
from src.candidate_selection import CandidateSelection
from sklearn.model_selection import train_test_split
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

    candidate_dataset = DataSet(
        candidate_df,meta_information=['feature1','label'],
        regime='supervised',label_column='label')

    safety_dataset = DataSet(
        safety_df,meta_information=['feature1','label'],
        regime='supervised',label_column='label')
    n_safety = len(safety_df)
    # Linear regression model
    from src.model import LinearRegressionModel
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
    cs = CandidateSelection(
        model=model_instance,
        candidate_dataset=candidate_dataset,
        n_safety=n_safety,
        parse_trees=parse_trees,
        primary_objective=model_instance.sample_Mean_Squared_Error,
        optimization_technique='barrier_function',
        optimizer='Powell',
        initial_solution_fn=model_instance.fit,
        minimizer_options=minimizer_options)
    candidate_solution = cs.run(minimizer_options=minimizer_options)
    assert candidate_solution[0] == pytest.approx(0.0280619)
    assert candidate_solution[1] == pytest.approx(0.95523116)
    
def test_GPA_data(generate_data):
    print()
    np.random.seed(0)
    csv_file = '../datasets/GPA/data_phil_modified.csv'
    columns = ["M","F","SAT_Physics",
           "SAT_Biology","SAT_History",
           "SAT_Second_Language","SAT_Geography",
           "SAT_Literature","SAT_Portuguese_and_Essay",
           "SAT_Math","SAT_Chemistry","GPA"]
    sensitive_column_names = ['M','F']
    loader = DataSetLoader(column_names=columns,
        sensitive_column_names=sensitive_column_names,
        regime='supervised',label_column='GPA')
    dataset = loader.from_csv(csv_file)

    candidate_df, safety_df = train_test_split(
            dataset.df, test_size=0.6, shuffle=False)

    candidate_dataset = DataSet(
        candidate_df,meta_information=dataset.df.columns,
        sensitive_column_names=sensitive_column_names,
        regime='supervised',label_column='GPA')
    
    n_safety = len(safety_df)

    # Linear regression model
    from src.model import LinearRegressionModel
    model_instance = LinearRegressionModel()

    # Constraints
    constraint_str = 'abs((Mean_Error | [M]) - (Mean_Error | [F])) - 0.05'
    delta = 0.05
    # Create parse tree object
    pt = ParseTree(delta=delta)

    pt.create_from_ast(constraint_str)
    
    # assign deltas for each base node
    # use equal weighting for each base node
    pt.assign_deltas(weight_method='equal')
    parse_trees = [pt]
    
    # # Candidate selection
    minimizer_options = {}

    cs = CandidateSelection(
        model=model_instance,
        candidate_dataset=candidate_dataset,
        n_safety=n_safety,
        parse_trees=parse_trees,
        primary_objective=model_instance.sample_Mean_Squared_Error,
        optimization_technique='barrier_function',
        optimizer='Powell',
        initial_solution_fn=model_instance.fit)

    candidate_solution = cs.run(minimizer_options=minimizer_options)
    # print(candidate_solution)


