""" Module containing global variables used 
during the construction of parse trees 

.. data:: measure_functions_dict
	:type: dict[regime][sub_regime]

	Contains strings that, if appearing in 
	a constraint string, will be recognized
	by the engine as statistical functions with 
	special meaning. Organized by regime and sub-regime. 
	For reference the meaning of each measure function is listed here:
		
		Supervised classification: 

		- 'PR': Positive rate
		- 'NR': Negative rate
		- 'FPR': False positive rate
		- 'TPR': True positive rate
		- 'FNR': False negative rate
		- 'TNR': True negative rate
		- 'logistic_loss': Logistic loss
		
		Supervised regression:
			
		- 'Mean_Error': Mean error
		- 'Mean_Squared_Error': Mean squared error
		
		Reinforcement learning:

		- 'J_pi_new': The performance (expected return of weighted rewards) of the new policy

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

.. data:: bounds_required_dict
	:type: dict

	Defines a map specifying which child bounds
	are required for each operator. If an operator
	has two children, A and B, then
	arrays are boolean of length 4, like: 
	[need_A_lower,need_A_upper,need_B_lower,need_B_upper]

	If an operator has one child, A, then
	arrays are boolean:
	[need_A_lower, need_A_upper]
"""
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
import ast

from .nodes import *


measure_functions_dict = {
	'supervised_learning': {
		'classification':
			[
			'PR',
			'NR',
			'FPR',
			'TPR',
			'FNR',
			'TNR',
			],
		'multiclass_classification':
			[
			'CM',
			'PR',
			'NR',
			'FPR',
			'TPR',
			'FNR',
			'TNR',
			],
		'regression':
			[
			'Mean_Error',
			'Mean_Squared_Error'
			]
		},
	'reinforcement_learning': {'all':['J_pi_new']}
}

custom_base_node_dict = {
	'MED_MF':MEDCustomBaseNode,
	'CVaRSQE':CVaRSQeBaseNode,
}

op_mapper = {
	ast.Sub: 'sub',
	ast.Add: 'add',
	ast.Mult:'mult',
	ast.Div: 'div',
	ast.Pow: 'pow'
}

not_supported_op_mapper = {
	ast.Mod: 'mod',
	ast.BitXor: '^',
	ast.LShift: '<<',
	ast.RShift: '>>',
	ast.BitAnd: '&',
	ast.FloorDiv: '//',
	ast.UAdd:'+',
	ast.Not:'not',
	ast.Invert:'~'
}

bounds_required_dict = {
	'add':{
	'lower':[1,0,1,0],
	'upper':[0,1,0,1],
	},
	'sub':{
	'lower':[1,0,0,1],
	'upper':[0,1,1,0],
	},
	'mult':{
	'lower':[1,1,1,1],
	'upper':[1,1,1,1],
	},
	'div':{
	'lower':[1,1,1,1],
	'upper':[1,1,1,1],
	},
	'pow':{
	'lower':[1,1,1,1],
	'upper':[1,1,1,1],
	},
	'min':{
	'lower':[1,0,1,0],
	'upper':[0,1,0,1],
	},
	'max':{
	'lower':[1,0,1,0],
	'upper':[0,1,0,1],
	},
	'abs':{
	'lower':[1,1],
	'upper':[1,1],
	},
	'exp':{
	'lower':[1,0],
	'upper':[0,1],
	},
}