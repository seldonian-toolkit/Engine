import sys
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from seldonian.dataset import (DataSetLoader,
	SupervisedDataSet)
from seldonian.parse_tree.parse_tree import ParseTree

if __name__ == '__main__':
	# gpa dataset
	np.random.seed(0)
	csv_file = '../datasets/GPA/data_classification_01_scaled.csv'
	columns = ["M","F","SAT_Physics",
		   "SAT_Biology","SAT_History",
		   "SAT_Second_Language","SAT_Geography",
		   "SAT_Literature","SAT_Portuguese_and_Essay",
		   "SAT_Math","SAT_Chemistry","GPA_class"]

	
	# Constraints

	# Demographic parity
	constraint_str = 'abs((PR | [M]) - (PR | [F])) - 0.2'

	# Predictive equality 
	# constraint_str = 'abs((FPR | [M]) - (FPR | [F])) - 0.2'

	# Equal opportunity 
	# constraint_str = 'abs((FNR | [M]) - (FNR | [F])) - 0.2'

	# Equalized odds - predictive equality and equal opportunity - 

	# Disparate impact 
	# constraint_str = '0.8 - min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M]))'

	delta = 0.05
	# Create parse tree object
	pt = ParseTree(delta=delta,regime='supervised',
		sub_regime='classification',columns=columns)
	
	pt.build_tree(constraint_str)
	
	title = f'Parse tree for expression:\n{constraint_str}\ndelta={delta}'
	graph = pt.make_viz(title)
	graph.attr(fontsize='12')
	graph.view() # to open it as a pdf
	# input("End of optimzer iteration")
	# Candidate selection