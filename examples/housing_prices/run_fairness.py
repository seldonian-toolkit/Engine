# loan fairness
import os

from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.io_utils import load_pickle

if __name__ == '__main__':
    # Load loan spec file
    specfile = './spec.pkl'
    spec = load_pickle(specfile)

    SA = SeldonianAlgorithm(spec)
    passed_safety,solution = SA.run(write_cs_logfile=True,debug=True)
    if passed_safety:
        print("Passed safety test!")
    else:
        print("Failed safety test")
