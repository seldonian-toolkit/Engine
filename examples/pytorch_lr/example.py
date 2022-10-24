import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd.extend import defvjp
from seldonian.models.models import (LinearRegressionModel)
#     PytorchLinearRegressionModel)
from seldonian.models.pytorch_model import PytorchLinearRegressionModel2
from seldonian.spec import SupervisedSpec
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.tutorial_utils import (
    make_synthetic_regression_dataset)
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)

if __name__ == "__main__":
    np.random.seed(0)
    num_points=1000  
    # 1. Define the data - X ~ N(0,1), Y ~ X + N(0,1)
    dataset = make_synthetic_regression_dataset(
        num_points=num_points)

    # 2. Create parse trees from the behavioral constraints 
    # constraint strings:
    constraint_strs = ['Mean_Squared_Error >= 1.25','Mean_Squared_Error <= 2.0',]
    # confidence levels: 
    deltas = [0.1,0.1] 

    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,deltas)

    # 3. Define the underlying machine learning model
    # model = LinearRegressionModel()
    model = PytorchLinearRegressionModel2(input_dim=1,output_dim=1)

    """4. Create a spec object, using some
    hidden defaults we won't worry about here
    """
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime='regression',
    )
    spec.optimization_hyperparams['lambda_init'] = np.array([0.5,0.5])

    # 5. Run seldonian algorithm using the spec object
    SA = SeldonianAlgorithm(spec)
    # passed_safety,solution = SA.run(write_cs_logfile=True,debug=True)
    passed_safety,solution = SA.run()
    print(passed_safety,solution)
