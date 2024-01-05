# run_engine.py
from seldonian.utils.io_utils import load_pickle
from seldonian.seldonian_algorithm import SeldonianAlgorithm

def initial_solution_fn(m,X,y):
    return m.fit(X,y)

if __name__ == '__main__':

    savename = "demographic_parity_addl_datasets_nodups.pkl"
    spec = load_pickle(savename)
            
    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run(debug=True,write_cs_logfile=True)