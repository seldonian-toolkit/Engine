import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
from seldonian.models.models import LinearRegressionModel
from seldonian.dataset import SupervisedDataSet
from seldonian.parse_tree.parse_tree import ParseTree
from seldonian.spec import SupervisedSpec
from seldonian.seldonian_algorithm import SeldonianAlgorithm


if __name__ == "__main__":
    np.random.seed(0)
    numPoints=1000

    # 1. Define the data
    def generate_data(numPoints,loc_X=0.0,
        loc_Y=0.0,sigma_X=1.0,sigma_Y=1.0):
        """ The function we will use to generate
        synthetic data
        """
        # Sample x from a standard normal distribution
        X = np.random.normal(loc_X, sigma_X, numPoints)
        # Set y to be x, plus noise from a standard normal distribution
        Y = X + np.random.normal(loc_Y, sigma_Y, numPoints)
        return (X,Y)
        
    X,Y = generate_data(numPoints)

    # 2. Define the metadataa
    columns = columns=['feature1','label']

    # 3. Make a dataset object
    rows = np.hstack([np.expand_dims(X,axis=1),
        np.expand_dims(Y,axis=1)])
    df = pd.DataFrame(rows,columns=columns)

    dataset = SupervisedDataSet(df,
        meta_information=columns,
        label_column='label',
        include_intercept_term=True)

    """ include_intercept_term=True
    adds a column of ones in the
    feature array for convenience
    during matrix multiplication.
    """

    # 4. Define the behavioral constraints
    constraint_strs = ['1.25 - Mean_Squared_Error','Mean_Squared_Error - 2.0']
    deltas = [0.1,0.1] # confidence levels

    # 5. Make the parse trees from these behavioral constraints

    parse_trees = []
    for ii in range(len(constraint_strs)):
        constraint_str = constraint_strs[ii]

        delta = deltas[ii]

        # Create parse tree object
        parse_tree = ParseTree(
            delta=delta,
            regime='supervised',
            sub_regime='regression',
            columns=columns)

        # Fill out tree
        parse_tree.create_from_ast(constraint_str)
        # assign deltas for each base node
        # use equal weighting for each unique base node
        parse_tree.assign_deltas(weight_method='equal')

        # Assign bounds needed on the base nodes
        parse_tree.assign_bounds_needed()

        parse_trees.append(parse_tree)

    # 6. Define the underlying machine learning model and primary objective
    model_class = LinearRegressionModel

    primary_objective = model_class().sample_Mean_Squared_Error

    # 7. Define initial solution function
    def initial_solution(X,y):
        """ Initial solution will be [0,0] """
        return np.zeros(2)

    initial_solution_fn=initial_solution

    # 8. Decide what fraction of your data to split into
    # candidate selection vs. the safety test.
    frac_data_in_safety=0.6

    """ 9. Decide what method to use for computing the
    high confidence upper bound on each :math:`g_{i}`."""
    bound_method='ttest'

    """10. Create a spec object, using some
    hidden defaults we won't worry about here"""
    spec = SupervisedSpec(
        dataset=dataset,
        model_class=model_class,
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        initial_solution_fn=initial_solution_fn,
        parse_trees=parse_trees,
        bound_method=bound_method,
    )

    # 11. Run seldonian algorithm using the spec object
    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run()
    print(passed_safety,solution)

    # Check the value of the primary objective on the candidate dataset
    cs_primary_objective = SA.evaluate_primary_objective(theta=solution,
        branch='candidate_selection')
    print(cs_primary_objective)

    # Check the value of the primary objective on the safety dataset
    st_primary_objective = SA.evaluate_primary_objective(theta=solution,
        branch='safety_test')
    print(st_primary_objective)
