import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
import ast

from seldonian.nodes import *

measure_functions = [
	'Mean_Error',
	'Mean_Squared_Error',
	'PR',
	'NR',
	'ER',
	'FPR',
	'TPR',
	'FNR',
	'TNR',
	'J_pi_new',
]

custom_base_node_dict = {
	'MED_MF':MEDCustomBaseNode
}

# map these supported ast operators
# to string representations of operators
op_mapper = {
	ast.Sub: 'sub',
	ast.Add: 'add',
	ast.Mult:'mult',
	ast.Div: 'div',
	ast.Mod: 'modulo',
	ast.Pow: 'pow'
}

# Not supported ast operators
not_supported_op_mapper = {
	ast.BitXor: '^',
	ast.LShift: '<<',
	ast.RShift: '>>',
	ast.BitAnd: '&',
	ast.FloorDiv: '//',
	ast.UAdd:'+',
	ast.Not:'not',
	ast.Invert:'~'
}
