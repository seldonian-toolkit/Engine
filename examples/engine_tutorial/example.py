import autograd.numpy as np   # Thinly-wrapped version of Numpy
from seldonian.models.models import LinearRegressionModel
from seldonian.spec import createSimpleSupervisedSpec
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.tutorial_utils import (
    make_synthetic_regression_dataset)

if __name__ == "__main__":
    np.random.seed(0)
    num_points=1000  
    # 1. Define the data - X ~ N(0,1), Y ~ X + N(0,1)
    dataset = make_synthetic_regression_dataset(
        num_points=num_points)

    # 2. Specify behavioral constraints 

    constraint_strs = ['Mean_Squared_Error >= 1.25',
        'Mean_Squared_Error <= 2.0']
    deltas = [0.1,0.1] # confidence levels

    # 3. Define the underlying machine learning model
    model = LinearRegressionModel()

    """4. Create a spec object, using some
    hidden defaults we won't worry about here
    """
    spec = createSimpleSupervisedSpec(
        dataset=dataset,
        constraint_strs=constraint_strs,
        deltas=deltas,
        sub_regime='regression',
    )

    # 5. Run seldonian algorithm using the spec object
    SA = SeldonianAlgorithm(spec)
    # print(SA.initial_solution)

    passed_safety,solution = SA.run()
    print(passed_safety,solution)

    st_primary_objective = SA.evaluate_primary_objective(
        theta=solution,
        branch='safety_test')
    print(st_primary_objective)

    cs_primary_objective = SA.evaluate_primary_objective(
    theta=solution,
    branch='candidate_selection')
    print(cs_primary_objective)

    print(SA.get_st_upper_bounds())

    cs_dict = SA.get_cs_result() # returns a dictionary with a lot of quantities evaluated at each step of gradient descent
    # print(list(cs_dict.keys()))
    # print(cs_dict['f_vals'])
    # print(cs_dict['g_vals'])