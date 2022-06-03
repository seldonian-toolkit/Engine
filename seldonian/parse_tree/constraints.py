""" Module containing global variables used 
during the construction of parse trees 

.. data:: measure_functions
	:type: List(str)

	A list of short-hand 
	function names that will be recognized to
	have a specific statistical meaning in 
	a constraint string provided in the interface.

.. data:: custom_base_node_dict
	:type: dict

	A dictionary mapping the name of a custom 
	base node as it would appear in the 
	constraint string to the class representing it 
	in :py:mod:`.nodes`

.. data:: op_mapper
	:type: dict

	Maps the supported ast operators
	to string representations of those operators
	as they appear in behavioral constraint strings

.. data:: not_supported_op_mapper
	:type: dict

	Not supported ast operators, mapped to 
	string representations of those operators
	as they appear in behavioral constraint strings
"""
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
import ast

from .nodes import *


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
	'logistic_loss'
]

custom_base_node_dict = {
	'MED_MF':MEDCustomBaseNode
}

op_mapper = {
	ast.Sub: 'sub',
	ast.Add: 'add',
	ast.Mult:'mult',
	ast.Div: 'div',
	ast.Mod: 'modulo',
	ast.Pow: 'pow'
}

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
