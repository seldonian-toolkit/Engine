import autograd.numpy as np
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.io_utils import load_pickle

if __name__ == '__main__':
    # load specfile
    specfile = './spec.pkl'
    spec = load_pickle(specfile)
    
    spec.optimization_hyperparams['num_iters']=10
    spec.optimization_hyperparams['alpha_theta']=0.05
    spec.optimization_hyperparams['alpha_lamb']=0.05
    # Run Seldonian algorithm 
    spec.frac_data_in_safety = 0.5
    # spec.initial_solution_fn = lambda x: np.array([
    #         [ 0.32139595, -0.32150993,  0.31890747, -0.31550895, -0.32887649,],
    #         [ 0.3243031,  -0.31302794,  0.31823579, -0.31966701,  0.31207868,],
    #         [-0.32055134,  0.32356196, -0.31955357,  0.31480999,  0.31859902,]])
    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run(debug=True,write_cs_logfile=True)
    if passed_safety:
        print("Passed safety test!")
        print("The solution found is:")
        print(solution)
    else:
        print("Failed safety test")
        print("No Solution Found")
    # print(SA.evaluate_primary_objective(branch='candidate_selection',theta=solution))
    # print(SA.evaluate_primary_objective(branch='safety_test',theta=solution))
