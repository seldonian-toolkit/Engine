import pytest
import time

from sklearn.model_selection import train_test_split

from seldonian.parse_tree.parse_tree import *
from seldonian.dataset import (DataSetLoader,
	SupervisedDataSet)
from seldonian.safety_test.safety_test import SafetyTest
from seldonian.utils.io_utils import load_json
from seldonian.models.models import LinearRegressionModel


two_interval_options = [
	[[2.0,3.0],[4.0,5.0]],
	[[-2.0,1.0],[2.0,3.0]],
	[[0.5,0.75],[2,4]],
	[[0.5,1.1],[0.0,0.4]],
	[[-3.2,-2.0],[-4.5,-4.0]],
	[[-3.2,-2.0],[-4.5,5.0]],
	[[-6.1,-6.0],[0.0,0.5]],
	[[0.0,0.5],[1.0,9.0]],
	[[0.0,0.5],[-6.8,-5.0]],
	[[float('-inf'),5.0],[-6.8,-5.0]],
	[[0.0,float('inf')],[5.0,10.0]],
	[[float('-inf'),float('inf')],[5.0,10.0]],
	[[float('-inf'),float('inf')],[float('-inf'),float('inf')]],
	[[float('inf'),float('inf')],[float('inf'),float('inf')]],
]

single_interval_options = [
	[-3.2,-2.0],
	[-3.2,2.0],
	[-5.1,0],
	[0.0,0.5],
	[float('-inf'),0.0],
	[float('-inf'),15342],
	[0.0,float('inf')],
	[float('-inf'),float('inf')],
	[float('inf'),float('inf')]
]

answer_dict = {
	'add': [
		[6.0,8.0],
		[0.0,4.0],
		[2.5,4.75],
		[0.5,1.5],
		[-7.7, -6.0],
		[-7.7,3.0],
		[-6.1, -5.5],
		[1.0, 9.5],
		[-6.8, -4.5],
		[float('-inf'), 0.0],
		[5.0, float('inf')],
		[float('-inf'), float('inf')],
		[float('-inf'), float('inf')],
		[float('inf'), float('inf')],
	],
	'sub': [
		[-3.0,-1.0],
		[-5.0,-1.0],
		[-3.5,-1.25],
		[0.1,1.1],
		[0.8, 2.5],
		[-8.2, 2.5],
		[-6.6, -6.0],
		[-9.0, -0.5],
		[5.0, 7.3],
		[float('-inf'), 11.8],
		[-10.0, float('inf')],
		[float('-inf'), float('inf')],
		[float('-inf'), float('inf')],
		[float('-inf'), float('inf')],
	],
	'mult': [
		[8.0,15.0],
		[-6.0,3.0],
		[1.0,3.0],
		[0.0,1.1*0.4],
		[8.0, 14.4],
		[-16.0, 14.4],
		[-3.05, 0.0],
		[0.0, 4.5],
		[-3.4, 0.0],
		[-34.0, float('inf')],
		[0.0, float('inf')],
		[float('-inf'), float('inf')],
		[float('-inf'), float('inf')],
		[float('inf'), float('inf')],
	],

	'div': [
		[2/5.0,3/4.0],
		[-1.0,0.5],
		[1/8,3/8],
		[0.5/0.4,float('inf')],
		[2/4.5, 3.2/4],
		[float('-inf'),float('inf')],
		[float('-inf'), -12.0],
		[0.0, 0.5],
		[-0.1, 0.0],
		[-1.0, float('inf')],
		[0.0, float('inf')],
		[float('-inf'), float('inf')],
		[float('-inf'), float('inf')],
		[float('-inf'), float('inf')],
	],
	'pow':[
		[16,243],
		[None,None],
		[pow(0.5,4),pow(0.75,2)],
		[pow(0.5,0.4),pow(1.1,0.4)],
		[None,None], # input raises exception 
		[None,None], # input raises exception 
		[None,None], # input raises exception 
		[0.0, 0.5],
		[0.0, 0.5],
		[None,None], # input raises exception 
		[0.0,float('inf')],
		[None,None], # input raises exception 
		[None,None], # input raises exception 
		[float('inf'),float('inf')]
	],
	'min': [
		[2.0,3.0],
		[-2.0,1.0],
		[0.5,0.75],
		[0.0,0.4],
		[-4.5,-4.0],
		[-4.5,-2.0],
		[-6.1,-6.0],
		[0.0,0.5],
		[-6.8,-5.0],
		[float('-inf'),-5.0],
		[0.0,10.0],
		[float('-inf'),10.0],
		[float('-inf'),float('inf')],
		[float('inf'),float('inf')]
	],
	'max': [
		[4.0,5.0],
		[2.0,3.0],
		[2.0,4.0],
		[0.5,1.1],
		[-3.2,-2.0],
		[-3.2,5.0],
		[0.0,0.5],
		[1.0,9.0],
		[0.0,0.5],
		[-6.8,5.0],
		[5.0,float('inf')],
		[5.0,float('inf')],
		[float('-inf'),float('inf')],
		[float('inf'),float('inf')]
	],
	'abs': [
		[2.0,3.2],
		[0,3.2],
		[0,5.1],
		[0,0.5],
		[0,float('inf')],
		[0,float('inf')],
		[0,float('inf')],
		[0,float('inf')],
		[float('inf'),float('inf')]
	]

}

### Begin tests

########################
### Propagator tests ###
########################

@pytest.mark.parametrize('interval_index',range(len(two_interval_options)))
def test_add_bounds(interval_index,stump):
	### Addition ###

	a,b=two_interval_options[interval_index]
	answer = answer_dict['add'][interval_index]
	pt = stump('add',a,b)
	pt.propagate_bounds()
	assert pt.root.lower == answer[0]
	assert pt.root.upper == answer[1]
	assert pt.base_node_dict['a']['bound_computed'] == True
	assert pt.base_node_dict['b']['bound_computed'] == True

@pytest.mark.parametrize('interval_index',range(len(two_interval_options)))
def test_subtract_bounds(interval_index,stump):
	### Subtraction ###

	a,b=two_interval_options[interval_index]
	answer = answer_dict['sub'][interval_index]
	pt = stump('sub',a,b)
	pt.propagate_bounds()
	# Use approx due to floating point imprecision
	assert pt.root.lower == pytest.approx(answer[0])
	assert pt.root.upper == pytest.approx(answer[1])
	assert pt.base_node_dict['a']['bound_computed'] == True
	assert pt.base_node_dict['b']['bound_computed'] == True

@pytest.mark.parametrize('interval_index',range(len(two_interval_options)))
def test_multiply_bounds(interval_index,stump):
	### Multiplication ###

	a,b=two_interval_options[interval_index]
	answer = answer_dict['mult'][interval_index]
	pt = stump('mult',a,b)
	pt.propagate_bounds()
	# Use approx due to floating point imprecision
	assert pt.root.lower == pytest.approx(answer[0])
	assert pt.root.upper == pytest.approx(answer[1])
	assert pt.base_node_dict['a']['bound_computed'] == True
	assert pt.base_node_dict['b']['bound_computed'] == True

@pytest.mark.parametrize('interval_index',range(len(two_interval_options)))
def test_divide_bounds(interval_index,stump):
	### Division ###

	a,b=two_interval_options[interval_index]
	answer = answer_dict['div'][interval_index]
	pt = stump('div',a,b)
	pt.propagate_bounds()
	# Use approx due to floating point imprecision
	assert pt.root.lower == pytest.approx(answer[0])
	assert pt.root.upper == pytest.approx(answer[1])
	assert pt.base_node_dict['a']['bound_computed'] == True
	assert pt.base_node_dict['b']['bound_computed'] == True

@pytest.mark.parametrize('interval_index',range(len(two_interval_options)))
def test_power_bounds(interval_index,stump):
	### power ###

	# A warning message should be raised 
	# anytime the power operator is called
	warning_msg = ("Warning: Power operation "
		"is an experimental feature. Use with caution.")
	a,b=two_interval_options[interval_index]

	pt = stump('pow',a,b)
	if a[0] < 0:
		with pytest.warns(UserWarning,match=warning_msg):
			with pytest.raises(ArithmeticError) as excinfo:
				pt.propagate_bounds()
		
		assert "Cannot compute interval" in str(excinfo.value)
		assert "because first argument contains negatives" in str(excinfo.value)

	elif 0 in a and (b[0]<0 or b[1]<1):
		with pytest.warns(UserWarning,match=warning_msg):
			with pytest.raises(ZeroDivisionError) as excinfo:
				pt.propagate_bounds()
		
		assert "0.0 cannot be raised to a negative power" in str(excinfo.value)
	else:
		answer = answer_dict['pow'][interval_index]

		with pytest.warns(UserWarning,match=warning_msg):
			pt.propagate_bounds()
		
		# Use approx due to floating point imprecision
		assert pt.root.lower == pytest.approx(answer[0])
		assert pt.root.upper == pytest.approx(answer[1])
		assert pt.base_node_dict['a']['bound_computed'] == True
		assert pt.base_node_dict['b']['bound_computed'] == True

@pytest.mark.parametrize('interval_index',range(len(two_interval_options)))
def test_min_bounds(interval_index,stump):
	### min ###

	a,b=two_interval_options[interval_index]

	pt = stump('min',a,b)
	
	answer = answer_dict['min'][interval_index]

	pt.propagate_bounds()
	# Use approx due to floating point imprecision
	assert pt.root.lower == pytest.approx(answer[0])
	assert pt.root.upper == pytest.approx(answer[1])
	assert pt.base_node_dict['a']['bound_computed'] == True
	assert pt.base_node_dict['b']['bound_computed'] == True

@pytest.mark.parametrize('interval_index',range(len(two_interval_options)))
def test_max_bounds(interval_index,stump):
	### min ###

	a,b=two_interval_options[interval_index]

	pt = stump('max',a,b)
	
	answer = answer_dict['max'][interval_index]

	pt.propagate_bounds()
	# Use approx due to floating point imprecision
	assert pt.root.lower == pytest.approx(answer[0])
	assert pt.root.upper == pytest.approx(answer[1])
	assert pt.base_node_dict['a']['bound_computed'] == True
	assert pt.base_node_dict['b']['bound_computed'] == True

@pytest.mark.parametrize('interval_index',range(len(single_interval_options)))
def test_abs_bounds(interval_index,edge):
	### Absolute value ###

	a=single_interval_options[interval_index]
	answer = answer_dict['abs'][interval_index]
	pt = edge('abs',a)
	pt.propagate_bounds()
	# Use approx due to floating point imprecision
	assert pt.root.lower == pytest.approx(answer[0])
	assert pt.root.upper == pytest.approx(answer[1])
	assert pt.base_node_dict['a']['bound_computed'] == True

##################
### Node tests ###
##################

def test_node_reprs(stump):
	a,b=[[2.0,3.0],[4.0,5.0]]

	pt = stump('add',a,b)
	
	pt.assign_deltas()
	pt.propagate_bounds()
	
	# Before assigning which bounds are needed
	root_bounds_str = f"[6, 8]"
	assert pt.root.__repr__() == '\n'.join(
		["[0]","add",u'\u03B5' + ' ' + root_bounds_str])
	left_bounds_str = f"[2, 3]"
	assert pt.root.left.__repr__() == '\n'.join(
		["[1]","a",u'\u03B5' + ' ' + left_bounds_str + u', \u03B4=0.025'])
	
	right_bounds_str = f"[4, 5]"
	assert pt.root.right.__repr__() == '\n'.join(
		["[2]","b",u'\u03B5' + ' ' + right_bounds_str + u', \u03B4=0.025'])
	# After assigning which bounds are needed 
	pt = stump('add',a,b)
	
	pt.assign_deltas()
	pt.assign_bounds_needed()
	pt.propagate_bounds()
	
	# Before assigning which bounds are needed
	root_bounds_str = f"[_, 8]"
	assert pt.root.__repr__() == '\n'.join(
		["[0]","add",u'\u03B5' + ' ' + root_bounds_str])
	left_bounds_str = f"[_, 3]"
	assert pt.root.left.__repr__() == '\n'.join(
		["[1]","a",u'\u03B5' + ' ' + left_bounds_str + u', \u03B4=0.025'])
	
	right_bounds_str = f"[_, 5]"
	assert pt.root.right.__repr__() == '\n'.join(
		["[2]","b",u'\u03B5' + ' ' + right_bounds_str + u', \u03B4=0.025'])

########################
### Parse tree tests ###
########################

def test_parse_tree_from_simple_string():
	constraint_str = 'FPR - (FNR + PR)*4'
	delta = 0.05
	pt = ParseTree(delta,
		regime='supervised_learning',sub_regime='classification')
	pt.create_from_ast(constraint_str)
	assert pt.n_nodes == 7
	assert pt.n_base_nodes == 3
	assert len(pt.base_node_dict) == 3
	assert isinstance(pt.root,InternalNode)
	assert pt.root.name == 'sub'
	assert pt.root.left.name == 'FPR'
	assert pt.root.right.name == 'mult'
	assert pt.root.right.left.name == 'add'
	assert pt.root.right.left.left.name == 'FNR'
	assert pt.root.right.left.right.name == 'PR'
	assert pt.root.right.right.name == '4'
	assert pt.root.right.right.value == 4

def test_parse_tree_with_inequalities():
	# First one without inequalities
	# constraint_str = 'FPR <= 0.5 + 0.3*(PR | [M])'
	constraint_str = 'FPR - (0.5 + (PR | [M]))'
	pt = ParseTree(delta=0.05,regime='supervised_learning',
			sub_regime='classification',columns=['M'])

	# Fill out tree
	pt.build_tree(
		constraint_str=constraint_str,
		delta_weight_method='equal')
	
	assert pt.n_nodes == 5
	assert pt.n_base_nodes == 2
	assert len(pt.base_node_dict) == 2
	assert isinstance(pt.root,InternalNode)
	assert pt.root.name == 'sub'
	assert pt.root.left.name == 'FPR'
	assert pt.root.right.name == 'add'
	assert pt.root.right.left.name == '0.5'
	assert pt.root.right.left.value == 0.5
	assert pt.root.right.right.name == 'PR | [M]'

	# Now with <= 
	constraint_str_lte = 'FPR <= 0.5 + (PR | [M])'
	pt_lte = ParseTree(delta=0.05,regime='supervised_learning',
			sub_regime='classification',columns=['M'])

	# Fill out tree
	pt_lte.build_tree(
		constraint_str=constraint_str_lte,
		delta_weight_method='equal')
	
	assert pt_lte.n_nodes == 5
	assert pt_lte.n_base_nodes == 2
	assert len(pt_lte.base_node_dict) == 2
	assert isinstance(pt_lte.root,InternalNode)
	assert pt_lte.root.name == 'sub'
	assert pt_lte.root.left.name == 'FPR'
	assert pt_lte.root.right.name == 'add'
	assert pt_lte.root.right.left.name == '0.5'
	assert pt_lte.root.right.left.value == 0.5
	assert pt_lte.root.right.right.name == 'PR | [M]'

	# Now with >= 
	constraint_str_gte = '0.5 + (PR | [M]) >= FPR'
	pt_gte = ParseTree(delta=0.05,regime='supervised_learning',
			sub_regime='classification',columns=['M'])

	# Fill out tree
	pt_gte.build_tree(
		constraint_str=constraint_str_gte,
		delta_weight_method='equal')
	
	assert pt_gte.n_nodes == 5
	assert pt_gte.n_base_nodes == 2
	assert len(pt_gte.base_node_dict) == 2
	assert isinstance(pt_gte.root,InternalNode)
	assert pt_gte.root.name == 'sub'
	assert pt_gte.root.left.name == 'FPR'
	assert pt_gte.root.right.name == 'add'
	assert pt_gte.root.right.left.name == '0.5'
	assert pt_gte.root.right.left.value == 0.5
	assert pt_gte.root.right.right.name == 'PR | [M]'

	# <= 0
	constraint_str_lte0 = 'FPR - (0.5 + (PR | [M])) <= 0'
	pt_lte0 = ParseTree(delta=0.05,regime='supervised_learning',
			sub_regime='classification',columns=['M'])

	# Fill out tree
	pt_lte0.build_tree(
		constraint_str=constraint_str_lte0,
		delta_weight_method='equal')

	assert pt_lte0.n_nodes == 5
	assert pt_lte0.n_base_nodes == 2
	assert len(pt_lte0.base_node_dict) == 2
	assert isinstance(pt_lte0.root,InternalNode)
	assert pt_lte0.root.name == 'sub'
	assert pt_lte0.root.left.name == 'FPR'
	assert pt_lte0.root.right.name == 'add'
	assert pt_lte0.root.right.left.name == '0.5'
	assert pt_lte0.root.right.left.value == 0.5
	assert pt_lte0.root.right.right.name == 'PR | [M]'
	
	# >= 0
	constraint_str_gte0 = '0 >= FPR - (0.5 + (PR | [M]))'
	pt_gte0 = ParseTree(delta=0.05,regime='supervised_learning',
			sub_regime='classification',columns=['M'])

	# Fill out tree
	pt_gte0.build_tree(
		constraint_str=constraint_str_gte0,
		delta_weight_method='equal')

	assert pt_gte0.n_nodes == 5
	assert pt_gte0.n_base_nodes == 2
	assert len(pt_gte0.base_node_dict) == 2
	assert isinstance(pt_gte0.root,InternalNode)
	assert pt_gte0.root.name == 'sub'
	assert pt_gte0.root.left.name == 'FPR'
	assert pt_gte0.root.right.name == 'add'
	assert pt_gte0.root.right.left.name == '0.5'
	assert pt_gte0.root.right.left.value == 0.5
	assert pt_gte0.root.right.right.name == 'PR | [M]'


def test_math_functions():
	""" Test that math functions like
	min(), max(), abs() and exp() get parsed 
	as expected. min and max expect no more than two arguments."""
	constraint_str = 'min((PR | [X]), (PR | [Y]))'
	delta = 0.05
	pt = ParseTree(delta,
		regime='supervised_learning',
		sub_regime='classification',
		columns=['X','Y'])
	pt.create_from_ast(constraint_str)

	assert pt.n_nodes == 3
	assert pt.n_base_nodes == 2
	assert len(pt.base_node_dict) == 2
	assert isinstance(pt.root,InternalNode)
	assert pt.root.name == 'min'
	assert pt.root.left.name == 'PR | [X]'
	assert pt.root.right.name == 'PR | [Y]'

	constraint_str = 'min((PR | [X]), (PR | [Y]), (PR | [Z]))'
	delta = 0.05
	pt = ParseTree(delta,
		regime='supervised_learning',
		sub_regime='classification',
		columns=['X','Y','Z'])
	with pytest.raises(RuntimeError) as excinfo:
		pt.create_from_ast(constraint_str)
	
	error_str = ("Please check the syntax of the function: min()."
			 " It appears you provided more than two arguments")
	assert str(excinfo.value) == error_str

	# constraint_str = 'abs((PR | [X]), (PR | [Y]))'
	# delta = 0.05
	# pt = ParseTree(delta,
	# 	regime='supervised_learning',
	# 	sub_regime='classification',
	# 	columns=['X','Y','Z'])
	# # with pytest.raises(RuntimeError) as excinfo:
	# pt.create_from_ast(constraint_str)
	
	# error_str = ("Please check the syntax of the function: min()."
	# 		 " It appears you provided more than two arguments")
	# assert str(excinfo.value) == error_str

def test_measure_functions_recognized():
	delta = 0.05

	constraint_str = 'Mean_Squared_Error - 2.0'
	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression')
	pt.create_from_ast(constraint_str)
	assert pt.root.left.measure_function_name == 'Mean_Squared_Error'
	
	constraint_str = '(Mean_Error|[M]) - 2.0'

	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression',columns=['M'])
	pt.create_from_ast(constraint_str)
	assert pt.root.left.measure_function_name == 'Mean_Error'

	constraint_str = '(FPR|[A,B]) - 2.0'

	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='classification',columns=['A','B'])
	pt.create_from_ast(constraint_str)
	assert pt.root.left.measure_function_name == 'FPR'

	# Test that a non-measure base node 
	# is not recognized as measure
	constraint_str = 'X - 2.0'

	pt = ParseTree(delta,regime='supervised_learning',sub_regime='classification')
	with pytest.raises(NotImplementedError) as excinfo:
		pt.create_from_ast(constraint_str)
	
	error_str = ("Error parsing your expression."
			 " A variable name was used which we do not recognize: X")
	assert str(excinfo.value) == error_str

	# Test that a non-measure base node 
	# is not recognized as measure
	constraint_str = '(X | [A]) - 2.0'

	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='classification',columns=['A'])
	with pytest.raises(NotImplementedError) as excinfo:
		pt.create_from_ast(constraint_str)
	
	error_str = ("Error parsing your expression."
			 " A variable name was used which we do not recognize: X")
	assert str(excinfo.value) == error_str
 
def test_measure_function_with_conditional_bad_syntax_captured():
	delta=0.05
	error_str = ("Error parsing your expression."
				" The issue is most likely due to"
				" missing/mismatched parentheses or square brackets"
				" in a conditional expression involving '|'.")

	bad_constraint_strs = [
		'Mean_Error | M ',
		'(Mean_Error | M)',
		'(Mean_Error | M,F)',
		'abs(Mean_Error | [M] - Mean_Error | [F]) - 0.1',
		'abs((Mean_Error | M) - (Mean_Error | F)) - 0.1',
		'abs((Mean_Error | [M]) - (Mean_Error | F)) - 0.1',
		'abs((Mean_Error | M) - (Mean_Error | [F])) - 0.1',
		'abs((Mean_Error | [M]) - (Mean_Error | F,L)) - 0.1',
		'abs((Mean_Error | A,B) - (Mean_Error | [F])) - 0.1',
		]
	
	for constraint_str in bad_constraint_strs:
		pt = ParseTree(delta,regime='supervised_learning',
			sub_regime='regression',columns=['A','B','M','F','L'])
		with pytest.raises(RuntimeError) as excinfo:
			pt.create_from_ast(constraint_str)
		
		assert str(excinfo.value) == error_str
 
def test_measure_function_from_wrong_regime():
	""" Test that if a measure function from the incorrect 
	regime or sub-regime is used in a constraint 
	that the parse tree builder will raise an error """
	delta = 0.05

	constraint_str = 'Mean_Squared_Error - 2.0'

	pt = ParseTree(delta,regime='supervised_learning',sub_regime='classification')

	with pytest.raises(NotImplementedError) as excinfo:
		pt.create_from_ast(constraint_str)
	
	error_str = ("NotImplementedError: Error parsing your expression. "
		"A variable name was used which we do not recognize: "
		"Mean_Squared_Error")
	assert str(excinfo.value) in error_str

	constraint_str = 'FPR - 0.2'

	pt = ParseTree(delta,regime='supervised_learning',sub_regime='regression')

	with pytest.raises(NotImplementedError) as excinfo:
		pt.create_from_ast(constraint_str)
	
	error_str = ("NotImplementedError: Error parsing your expression. "
		"A variable name was used which we do not recognize: "
		"FPR")
	assert str(excinfo.value) in error_str

	constraint_str = 'Mean_Squared_Error - 2.0'

	pt = ParseTree(delta,regime='reinforcement_learning',sub_regime='all')

	with pytest.raises(NotImplementedError) as excinfo:
		pt.create_from_ast(constraint_str)
	
	error_str = ("NotImplementedError: Error parsing your expression. "
		"A variable name was used which we do not recognize: "
		"Mean_Squared_Error")
	assert str(excinfo.value) in error_str

	constraint_str = '(FPR | [M]) - 0.2'

	pt = ParseTree(delta,regime='reinforcement_learning',sub_regime='all',columns=['M'])

	with pytest.raises(NotImplementedError) as excinfo:
		pt.create_from_ast(constraint_str)
	
	error_str = ("NotImplementedError: Error parsing your expression. "
		"A variable name was used which we do not recognize: "
		"FPR")
	assert str(excinfo.value) in error_str
 
def test_custom_base_nodes():
	constraint_str = 'MED_MF - 0.1'
	delta = 0.05 

	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression')
	pt.create_from_ast(constraint_str)
	assert isinstance(pt.root.left,BaseNode)
	assert isinstance(pt.root.left,MEDCustomBaseNode)
	assert pt.n_base_nodes == 1
	assert len(pt.base_node_dict) == 1

	constraint_str = 'CVARSQE - 1.0'
	delta = 0.05 

	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression')
	pt.create_from_ast(constraint_str)
	assert isinstance(pt.root.left,BaseNode)
	assert isinstance(pt.root.left,CVARSQEBaseNode)
	assert pt.root.left.alpha == 0.1
	assert pt.n_base_nodes == 1
	assert len(pt.base_node_dict) == 1

def test_unary_op():
	delta = 0.05 

	constraint_str = '-10+abs(Mean_Error)'
	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression')
	pt.create_from_ast(constraint_str)
	assert pt.root.name == 'add'
	assert pt.root.left.value == -10
	assert pt.root.left.name == '-10'
	assert pt.root.right.name == 'abs'
	assert pt.root.right.left.name == 'Mean_Error'
	assert pt.n_nodes == 4
	assert pt.n_base_nodes == 1

	constraint_str = '-MED_MF'
	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression')
	pt.create_from_ast(constraint_str)
	assert pt.root.name == 'mult'
	assert pt.root.left.value == -1
	assert pt.root.right.name == 'MED_MF'
	assert pt.n_nodes == 3


	constraint_str = '-abs(Mean_Error) - 10'
	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression')
	pt.create_from_ast(constraint_str)
	assert pt.root.name == 'sub'
	assert pt.root.right.value == 10
	assert pt.root.left.name == 'mult'
	assert pt.root.left.left.value == -1
	assert pt.root.left.right.name == 'abs'
	assert pt.root.left.right.left.name == 'Mean_Error'
	assert pt.n_nodes == 6
	assert pt.n_base_nodes == 1
	assert len(pt.base_node_dict) == 1

	constraint_str = '-abs(Mean_Error | [M]) - 10'
	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression',columns=['M'])
	pt.create_from_ast(constraint_str)
	assert pt.root.name == 'sub'
	assert pt.root.right.value == 10
	assert pt.root.left.name == 'mult'
	assert pt.root.left.left.value == -1
	assert pt.root.left.right.name == 'abs'
	assert pt.root.left.right.left.name == 'Mean_Error | [M]'
	assert pt.n_nodes == 6
	assert pt.n_base_nodes == 1
	assert len(pt.base_node_dict) == 1

def test_raise_error_on_excluded_operators():

	constraint_str = 'FPR^4'
	delta = 0.05
	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='classification')
	with pytest.raises(NotImplementedError) as excinfo:
		pt.create_from_ast(constraint_str)
	error_str = ("Error parsing your expression."
		 " An operator was used which we do not support: ^")
	assert str(excinfo.value) == error_str

	constraint_str = 'FPR<4'
	delta = 0.05
	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='classification')
	with pytest.raises(NotImplementedError) as excinfo:
		pt.create_from_ast(constraint_str)
	error_str = ("Error parsing your expression."
		 " An operator was used which we do not support: <")
	assert str(excinfo.value) == error_str

	constraint_str = 'FPR>4'
	delta = 0.05
	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='classification')
	with pytest.raises(NotImplementedError) as excinfo:
		pt.create_from_ast(constraint_str)
	error_str = ("Error parsing your expression."
		 " An operator was used which we do not support: >")
	assert str(excinfo.value) == error_str

	constraint_str = 'FPR & FNR'
	delta = 0.05
	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='classification')
	with pytest.raises(NotImplementedError) as excinfo:
		pt.create_from_ast(constraint_str)
	error_str = ("Error parsing your expression."
		 " An operator was used which we do not support: &")
	assert str(excinfo.value) == error_str

	constraint_str = 'FPR//4'
	delta = 0.05
	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='classification')
	with pytest.raises(NotImplementedError) as excinfo:
		pt.create_from_ast(constraint_str)
	error_str = ("Error parsing your expression."
		 " An operator was used which we do not support: //")
	assert str(excinfo.value) == error_str
 
def test_single_conditional_columns_assigned():

	constraint_str = 'abs(Mean_Error|[X]) - 0.1'
	delta = 0.05
	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression',columns=['X'])
	pt.create_from_ast(constraint_str)
	assert pt.n_nodes == 4
	assert pt.n_base_nodes == 1
	assert len(pt.base_node_dict) == 1  
	assert pt.root.left.left.conditional_columns == ['X']

def test_multiple_conditional_columns_assigned():

	constraint_str = 'abs(Mean_Error|[X,Y,Z]) - 0.1'
	delta = 0.05
	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression',columns=['X','Y','Z'])
	pt.create_from_ast(constraint_str)
	assert pt.n_nodes == 4
	assert pt.n_base_nodes == 1
	assert len(pt.base_node_dict) == 1  
	assert pt.root.left.left.conditional_columns == ['X','Y','Z']

def test_deltas_assigned_equally():
	constraint_str = 'abs((Mean_Error|[M]) - (Mean_Error|[F])) - 0.1'
	delta = 0.05 

	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression',columns=['M','F'])
	pt.create_from_ast(constraint_str)
	pt.assign_deltas(weight_method='equal')
	assert pt.n_nodes == 6
	assert pt.n_base_nodes == 2
	assert len(pt.base_node_dict) == 2  
	assert isinstance(pt.root,InternalNode)
	assert pt.root.name == 'sub'  
	assert pt.root.left.left.left.delta == delta/len(pt.base_node_dict)
	assert pt.root.left.left.right.delta == delta/len(pt.base_node_dict)

def test_deltas_assigned_once_per_unique_basenode():
	""" Make sure that the delta assigned to each base node 
	is delta/number_of_unique_base_nodes, such that if a base
	node appears more than once it doesn't further dilute delta 
	"""
	constraint_str = '0.8 - min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M]))'
	delta = 0.05 

	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='classification',columns=['M','F'])
	pt.create_from_ast(constraint_str)
	pt.assign_deltas(weight_method='equal')
	assert pt.n_nodes == 9
	assert pt.n_base_nodes == 4
	assert len(pt.base_node_dict) == 2  
	# assert isinstance(pt.root,InternalNode)
	# assert pt.root.name == 'sub'  
	# assert pt.root.left.left.left.delta == delta/pt.n_base_nodes
	# assert pt.root.left.left.right.delta == delta/pt.n_base_nodes

def test_bounds_needed_assigned_correctly():
	delta = 0.05 # use for all trees below

	constraint_str = 'FPR'
	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='classification')
	pt.create_from_ast(constraint_str)
	# Before bounds assigned both should be True
	assert pt.root.will_lower_bound == True
	assert pt.root.will_upper_bound == True
	pt.assign_bounds_needed()
	# But after, we should find that only upper is needed
	assert pt.n_nodes == 1
	assert pt.n_base_nodes == 1  
	assert isinstance(pt.root,BaseNode)

	assert pt.root.will_lower_bound == False
	assert pt.root.will_upper_bound == True

	constraint_str = '(Mean_Error | [M]) - 0.1'
	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression',columns=['M'])
	pt.create_from_ast(constraint_str)
	# Before bounds assigned both should be True
	assert pt.root.left.name == 'Mean_Error | [M]'
	assert pt.root.left.will_lower_bound == True
	assert pt.root.left.will_upper_bound == True
	pt.assign_bounds_needed()
	# But after, we should find that only upper is needed
	assert pt.n_nodes == 3
	assert pt.n_base_nodes == 1  
	assert isinstance(pt.root.left,BaseNode)
	assert pt.root.left.will_lower_bound == False
	assert pt.root.left.will_upper_bound == True

	constraint_str = '2.0 - Mean_Squared_Error'
	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression')
	pt.create_from_ast(constraint_str)
	# Before bounds assigned both should be True
	assert pt.root.right.name == 'Mean_Squared_Error'
	assert pt.root.right.will_lower_bound == True
	assert pt.root.right.will_upper_bound == True
	pt.assign_bounds_needed()
	# But after, we should find that only lower is needed
	assert pt.n_nodes == 3
	assert pt.n_base_nodes == 1  
	assert isinstance(pt.root.right,BaseNode)
	assert pt.root.right.will_lower_bound == True
	assert pt.root.right.will_upper_bound == False

	constraint_str = 'abs((Mean_Error | [M]) - (Mean_Error | [F])) - 0.1'
	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression',columns=['M','F'])
	pt.create_from_ast(constraint_str)
	# Before bounds assigned both should be True
	assert pt.root.left.left.left.name == 'Mean_Error | [M]'
	assert pt.root.left.left.left.will_lower_bound == True
	assert pt.root.left.left.left.will_upper_bound == True
	assert pt.root.left.left.right.name == 'Mean_Error | [F]'
	assert pt.root.left.left.right.will_lower_bound == True
	assert pt.root.left.left.right.will_upper_bound == True
	pt.assign_bounds_needed()
	# # After, we should find that both base nodes need both still 
	assert pt.n_nodes == 6
	assert pt.n_base_nodes == 2
	assert pt.root.left.left.left.will_lower_bound == True
	assert pt.root.left.left.left.will_upper_bound == True
	assert pt.root.left.left.right.will_lower_bound == True
	assert pt.root.left.left.right.will_upper_bound == True

	constraint_str = '(FPR * FNR) - 0.25'
	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='classification')
	pt.create_from_ast(constraint_str)
	# Before bounds assigned both should be True
	assert pt.root.left.left.name == 'FPR'
	assert pt.root.left.right.name == 'FNR'
	
	assert pt.root.left.left.will_lower_bound == True
	assert pt.root.left.left.will_upper_bound == True
	assert pt.root.left.right.will_lower_bound == True
	assert pt.root.left.right.will_upper_bound == True
	pt.assign_bounds_needed()
	# After, we should find that both base nodes need both still
	assert pt.n_nodes == 5
	assert pt.n_base_nodes == 2  
	assert pt.root.left.left.will_lower_bound == True
	assert pt.root.left.left.will_upper_bound == True
	assert pt.root.left.right.will_lower_bound == True
	assert pt.root.left.right.will_upper_bound == True

	constraint_str = '(TPR - FPR) - 0.25'
	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='classification')
	pt.create_from_ast(constraint_str)
	# Before bounds assigned both should be True
	assert pt.root.left.left.name == 'TPR'
	assert pt.root.left.right.name == 'FPR'
	
	assert pt.root.left.left.will_lower_bound == True
	assert pt.root.left.left.will_upper_bound == True
	assert pt.root.left.right.will_lower_bound == True
	assert pt.root.left.right.will_upper_bound == True
	pt.assign_bounds_needed()
	# After, we should find that both base nodes need both still
	assert pt.n_nodes == 5
	assert pt.n_base_nodes == 2  
	assert pt.root.left.left.will_lower_bound == False
	assert pt.root.left.left.will_upper_bound == True
	assert pt.root.left.right.will_lower_bound == True
	assert pt.root.left.right.will_upper_bound == False

	constraint_str = '(FPR + FNR) - 0.25'
	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='classification')
	pt.create_from_ast(constraint_str)
	# Before bounds assigned both should be True
	assert pt.root.left.left.name == 'FPR'
	assert pt.root.left.right.name == 'FNR'
	
	assert pt.root.left.left.will_lower_bound == True
	assert pt.root.left.left.will_upper_bound == True
	assert pt.root.left.right.will_lower_bound == True
	assert pt.root.left.right.will_upper_bound == True
	pt.assign_bounds_needed()
	# After, we should find that both base nodes need both still
	assert pt.n_nodes == 5
	assert pt.n_base_nodes == 2  
	assert pt.root.left.left.will_lower_bound == False
	assert pt.root.left.left.will_upper_bound == True
	assert pt.root.left.right.will_lower_bound == False
	assert pt.root.left.right.will_upper_bound == True

def test_duplicate_base_nodes():
	constraint_str = 'FPR + 4/FPR - 2.0'
	delta = 0.05 

	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='classification')
	pt.create_from_ast(constraint_str)
	assert pt.n_base_nodes == 2 
	assert len(pt.base_node_dict) == 1 
	assert pt.base_node_dict['FPR']['bound_computed'] == False
	pt.base_node_dict['FPR']['bound_method'] = "random"
	pt.propagate_bounds()
	assert pt.base_node_dict['FPR']['bound_computed'] == True

def test_ttest_bound(generate_data):
	# dummy data for linear regression
	np.random.seed(0)
	numPoints=1000

	model_instance = LinearRegressionModel()
	X,Y = generate_data(
		numPoints,loc_X=0.0,loc_Y=0.0,sigma_X=1.0,sigma_Y=1.0)
	rows = np.hstack([np.expand_dims(X,axis=1),np.expand_dims(Y,axis=1)])
	df = pd.DataFrame(rows,columns=['feature1','label'])
	
	frac_data_in_safety=0.6
	candidate_df, safety_df = train_test_split(
			df, test_size=frac_data_in_safety, shuffle=False)
	
	candidate_dataset = SupervisedDataSet(candidate_df,
		meta_information=['feature1','label'],
		regime='supervised_learning',label_column='label',
		include_sensitive_columns=False,
		include_intercept_term=True)

	safety_dataset = SupervisedDataSet(safety_df,
		meta_information=['feature1','label'],
		regime='supervised_learning',label_column='label',
		include_sensitive_columns=False,
		include_intercept_term=True)
	
	# First, single sided bound (MSE only needs upper bound)
	constraint_str = 'Mean_Squared_Error - 2.0'
	delta = 0.05 

	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression')
	pt.create_from_ast(constraint_str)
	pt.assign_deltas(weight_method='equal')
	pt.assign_bounds_needed()
	
	assert pt.n_nodes == 3
	assert pt.n_base_nodes == 1
	assert len(pt.base_node_dict) == 1
	assert pt.root.name == 'sub'  
	assert pt.root.left.will_lower_bound == False
	assert pt.root.left.will_upper_bound == True
	theta = np.array([0,1])
	
	# Candidate selection
	pt.propagate_bounds(theta=theta,dataset=candidate_dataset,
		n_safety=len(safety_df),
		model=model_instance,
		branch='candidate_selection',
		regime='supervised_learning')
	assert pt.root.lower == float('-inf') # not bound_computed 
	assert pt.root.upper == pytest.approx(-0.932847)
	pt.reset_base_node_dict(reset_data=True)
	# Safety test
	pt.propagate_bounds(theta=theta,dataset=safety_dataset,
		model=model_instance,
		branch='safety_test',
		regime='supervised_learning')
	assert pt.root.lower == float('-inf') # not computed
	assert pt.root.upper == pytest.approx(-0.947693)

	# Next, two sided bound 
	constraint_str = 'abs(Mean_Squared_Error) - 2.0'
	delta = 0.05 

	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression')
	pt.create_from_ast(constraint_str)
	pt.assign_deltas(weight_method='equal')
	pt.assign_bounds_needed()
	
	assert pt.n_nodes == 4
	assert pt.n_base_nodes == 1
	assert len(pt.base_node_dict) == 1
	theta = np.array([0,1])
	
	# Candidate selection
	pt.propagate_bounds(theta=theta,dataset=candidate_dataset,
		n_safety=len(safety_df),
		model=model_instance,
		branch='candidate_selection',
		regime='supervised_learning')
	
	# assert pt.root.lower == float('-inf') # not bound_computed 
	assert pt.root.upper == pytest.approx(-0.900307)
	pt.reset_base_node_dict(reset_data=True)
	# Safety test
	pt.propagate_bounds(theta=theta,dataset=safety_dataset,
		model=model_instance,
		branch='safety_test',
		regime='supervised_learning')
	# assert pt.root.lower == float('-inf') # not computed
	assert pt.root.upper == pytest.approx(-0.930726)

def test_bad_bound_method(generate_data):
	# dummy data for linear regression
	np.random.seed(0)
	numPoints=1000

	model_instance = LinearRegressionModel()
	X,Y = generate_data(
		numPoints,loc_X=0.0,loc_Y=0.0,sigma_X=1.0,sigma_Y=1.0)
	rows = np.hstack([np.expand_dims(X,axis=1),np.expand_dims(Y,axis=1)])
	df = pd.DataFrame(rows,columns=['feature1','label'])
	
	frac_data_in_safety=0.6
	candidate_df, safety_df = train_test_split(
			df, test_size=frac_data_in_safety, shuffle=False)
	
	candidate_dataset = SupervisedDataSet(candidate_df,
		meta_information=['feature1','label'],
		regime='supervised_learning',label_column='label',
		include_sensitive_columns=False,
		include_intercept_term=True)

	safety_dataset = SupervisedDataSet(safety_df,
		meta_information=['feature1','label'],
		regime='supervised_learning',label_column='label',
		include_sensitive_columns=False,
		include_intercept_term=True)
	
	# First, single sided bound (MSE only needs upper bound)
	constraint_str = 'Mean_Squared_Error - 2.0'
	delta = 0.05 

	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression')
	pt.create_from_ast(constraint_str)
	pt.assign_deltas(weight_method='equal')
	pt.assign_bounds_needed()
	
	assert pt.n_nodes == 3
	assert pt.n_base_nodes == 1
	assert len(pt.base_node_dict) == 1
	assert pt.root.name == 'sub'  
	assert pt.root.left.will_lower_bound == False
	assert pt.root.left.will_upper_bound == True
	theta = np.array([0,1])
	
	# Candidate selection
	bound_method = 'bad-method'
	pt.base_node_dict['Mean_Squared_Error']['bound_method'] = bound_method
	with pytest.raises(NotImplementedError) as excinfo:
		pt.propagate_bounds(theta=theta,dataset=candidate_dataset,
			n_safety=len(safety_df),
			model=model_instance,
			branch='candidate_selection',
			regime='supervised_learning')
	
	error_str = (f"Bounding method {bound_method} is not supported")
	assert str(excinfo.value) == error_str

	pt.reset_base_node_dict(reset_data=True)
	pt.base_node_dict['Mean_Squared_Error']['bound_method'] = bound_method
	# Safety test
	with pytest.raises(NotImplementedError) as excinfo:
		pt.propagate_bounds(theta=theta,dataset=safety_dataset,
			model=model_instance,
			branch='safety_test',
			regime='supervised_learning')

	error_str = (f"Bounding method {bound_method} is not supported")
	assert str(excinfo.value) == error_str

	# Next, other side single sided bound (MSE only needs lower bound)
	constraint_str = '1.0 - Mean_Squared_Error'
	delta = 0.05 

	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression')
	pt.create_from_ast(constraint_str)
	pt.assign_deltas(weight_method='equal')
	pt.assign_bounds_needed()
	pt.base_node_dict['Mean_Squared_Error']['bound_method'] = bound_method
	theta = np.array([0,1])
	
	# Candidate selection
	with pytest.raises(NotImplementedError) as excinfo:
		pt.propagate_bounds(theta=theta,dataset=candidate_dataset,
			n_safety=len(safety_df),
			model=model_instance,
			branch='candidate_selection',
			regime='supervised_learning')
	
	error_str = (f"Bounding method {bound_method} is not supported")
	assert str(excinfo.value) == error_str

	pt.reset_base_node_dict(reset_data=True)
	pt.base_node_dict['Mean_Squared_Error']['bound_method'] = bound_method
	# Safety test
	with pytest.raises(NotImplementedError) as excinfo:
		pt.propagate_bounds(theta=theta,dataset=safety_dataset,
			model=model_instance,
			branch='safety_test',
			regime='supervised_learning')

	error_str = (f"Bounding method {bound_method} is not supported")
	assert str(excinfo.value) == error_str

	# Now, two sided bound on leaf node
	constraint_str = 'abs(Mean_Squared_Error) - 2.0'
	delta = 0.05 

	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression')
	pt.create_from_ast(constraint_str)
	pt.assign_deltas(weight_method='equal')
	pt.assign_bounds_needed()
	pt.base_node_dict['Mean_Squared_Error']['bound_method'] = bound_method
	
	# Candidate selection
	bound_method = 'bad-method'
	with pytest.raises(NotImplementedError) as excinfo:
		pt.propagate_bounds(theta=theta,dataset=candidate_dataset,
			n_safety=len(safety_df),
			model=model_instance,
			branch='candidate_selection',
			regime='supervised_learning')
	
	error_str = (f"Bounding method {bound_method} is not supported")
	assert str(excinfo.value) == error_str

	pt.reset_base_node_dict(reset_data=True)
	pt.base_node_dict['Mean_Squared_Error']['bound_method'] = bound_method
	# Safety test
	with pytest.raises(NotImplementedError) as excinfo:
		pt.propagate_bounds(theta=theta,dataset=safety_dataset,
			model=model_instance,
			branch='safety_test',
			regime='supervised_learning')

	error_str = (f"Bounding method {bound_method} is not supported")
	assert str(excinfo.value) == error_str

def test_evaluate_constraint(generate_data,gpa_classification_dataset):
	# Evaluate constraint mean, not the bound
	# test all of the statistics in all regimes

	### Regression 
	np.random.seed(0)
	numPoints=1000

	model_instance = LinearRegressionModel()
	X,Y = generate_data(
		numPoints,loc_X=0.0,loc_Y=0.0,sigma_X=1.0,sigma_Y=1.0)

	rows = np.hstack([np.expand_dims(X,axis=1),np.expand_dims(Y,axis=1)])
	
	df = pd.DataFrame(rows,columns=['feature1','label'])
	
	dataset = SupervisedDataSet(df,meta_information=['feature1','label'],
		regime='supervised_learning',label_column='label',
		include_sensitive_columns=False,
		include_intercept_term=True)
	
	# MSE
	constraint_str = 'Mean_Squared_Error - 2.0'
	delta = 0.05 

	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression')
	pt.create_from_ast(constraint_str)

	pt.assign_deltas(weight_method='equal')
	pt.assign_bounds_needed()
	assert pt.n_nodes == 3
	assert pt.n_base_nodes == 1
	assert len(pt.base_node_dict) == 1
	
	theta = np.array([0,1])
	pt.evaluate_constraint(theta=theta,dataset=dataset,
		model=model_instance,regime='supervised_learning',
		branch='safety_test')

	assert pt.root.value == pytest.approx(-1.06248)

	# Mean error
	constraint_str = 'Mean_Error - 2.0'
	delta = 0.05 

	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression')
	pt.create_from_ast(constraint_str)

	pt.assign_deltas(weight_method='equal')
	pt.assign_bounds_needed()
	assert pt.n_nodes == 3
	assert pt.n_base_nodes == 1
	assert len(pt.base_node_dict) == 1
	
	theta = np.array([0,1])
	pt.evaluate_constraint(theta=theta,dataset=dataset,
		model=model_instance,regime='supervised_learning',
		branch='safety_test')

	assert pt.root.value == pytest.approx(-2.013617)

	### Classification
	constraint_str = '(PR + NR + FPR + FNR + TPR + TNR + logistic_loss) - 10.0'
	constraint_strs = [constraint_str]
	deltas = [0.05]

	(dataset,model_class,
		primary_objective,parse_trees) = gpa_classification_dataset(
		constraint_strs=constraint_strs,
		deltas=deltas)
	model_instance = model_class()

	theta = np.zeros(10)
	pt = parse_trees[0]
	pt.evaluate_constraint(theta=theta,dataset=dataset,
		model=model_instance,regime='supervised_learning',
		branch='safety_test')
	assert pt.root.value == pytest.approx(-6.306852)
	
def test_reset_parse_tree():
	
	constraint_str = '(FPR + FNR) - 0.5'
	delta = 0.05 

	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='classification')
	pt.create_from_ast(constraint_str)
	pt.assign_deltas(weight_method='equal')
	assert pt.n_base_nodes == 2
	assert len(pt.base_node_dict) == 2
	assert pt.base_node_dict['FPR']['bound_computed'] == False
	assert pt.base_node_dict['FPR']['lower'] == float('-inf')
	assert pt.base_node_dict['FPR']['upper'] == float('inf')
	assert pt.base_node_dict['FPR']['bound_method'] == 'ttest' # the default
	assert pt.base_node_dict['FNR']['lower'] == float('-inf')
	assert pt.base_node_dict['FNR']['upper'] == float('inf')
	assert pt.base_node_dict['FNR']['bound_computed'] == False
	assert pt.base_node_dict['FNR']['bound_method'] == 'ttest' # the default

	pt.base_node_dict['FPR']['bound_method'] = 'random' 
	pt.base_node_dict['FNR']['bound_method'] = 'random' 

	# propagate bounds
	pt.propagate_bounds()
	assert len(pt.base_node_dict) == 2
	assert pt.base_node_dict['FPR']['bound_computed'] == True
	assert pt.base_node_dict['FNR']['bound_computed'] == True
	assert pt.base_node_dict['FPR']['lower'] >= 0
	assert pt.base_node_dict['FPR']['upper'] > 0
	assert pt.base_node_dict['FNR']['lower'] >= 0
	assert pt.base_node_dict['FNR']['upper'] > 0

	# # reset the node dict 
	pt.reset_base_node_dict()
	assert len(pt.base_node_dict) == 2
	assert pt.base_node_dict['FPR']['bound_computed'] == False
	assert pt.base_node_dict['FNR']['bound_computed'] == False
	assert pt.base_node_dict['FPR']['lower'] == float('-inf')
	assert pt.base_node_dict['FPR']['upper'] == float('inf')
	assert pt.base_node_dict['FNR']['lower'] == float('-inf')
	assert pt.base_node_dict['FNR']['upper'] == float('inf')

def test_single_conditional_columns_propagated():
	np.random.seed(0)
	data_pth = 'static/datasets/supervised/GPA/gpa_regression_dataset.csv'
	metadata_pth = 'static/datasets/supervised/GPA/metadata_regression.json'

	metadata_dict = load_json(metadata_pth)

	loader = DataSetLoader(
		regime='supervised_learning')

	dataset = loader.load_supervised_dataset(
		filename=data_pth,
		metadata_filename=metadata_pth,
		include_sensitive_columns=False,
		include_intercept_term=True,
		file_type='csv')

	model_instance = LinearRegressionModel()

	constraint_str = 'abs(Mean_Error|[M]) - 0.1'
	delta = 0.05
	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='regression',
		columns=dataset.df.columns)
	
	pt.create_from_ast(constraint_str)
	pt.assign_deltas(weight_method='equal')

	# propagate the bounds with example theta value
	# theta = np.hstack([np.array([0.0,0.0]),np.random.uniform(-0.05,0.05,10)])
	theta = np.random.uniform(-0.05,0.05,10)
	pt.propagate_bounds(theta=theta,dataset=dataset,
		model=model_instance,branch='safety_test',
		regime='supervised_learning')
	assert pt.root.lower == pytest.approx(61.9001779655)
	assert pt.root.upper == pytest.approx(62.1362236720)

	assert len(pt.base_node_dict["Mean_Error | [M]"]['data_dict']['features']) == 22335
	pt.reset_base_node_dict()
	
def test_build_tree():
	""" Test the convenience function that builds the tree,
	weights deltas, and assigns bounds all in one """

	constraint_str = '(FPR + FNR) - 0.5'
	delta = 0.05 

	pt = ParseTree(delta,regime='supervised_learning',
		sub_regime='classification')

	# build the tree the original way 
	pt.create_from_ast(constraint_str)
	pt.assign_deltas(weight_method='equal')
	assert pt.n_base_nodes == 2
	assert len(pt.base_node_dict) == 2
	assert pt.base_node_dict['FPR']['bound_computed'] == False
	assert pt.base_node_dict['FPR']['lower'] == float('-inf')
	assert pt.base_node_dict['FPR']['upper'] == float('inf')
	assert pt.base_node_dict['FNR']['lower'] == float('-inf')
	assert pt.base_node_dict['FNR']['upper'] == float('inf')
	assert pt.base_node_dict['FNR']['bound_computed'] == False

	##### build the tree with the convenience function
	pt2 = ParseTree(delta,regime='supervised_learning',
		sub_regime='classification') 
	pt2.build_tree(constraint_str=constraint_str,
		delta_weight_method='equal')
	assert pt2.n_base_nodes == 2
	assert len(pt2.base_node_dict) == 2
	assert pt2.base_node_dict['FPR']['bound_computed'] == False
	assert pt2.base_node_dict['FPR']['lower'] == float('-inf')
	assert pt2.base_node_dict['FPR']['upper'] == float('inf')
	assert pt2.base_node_dict['FNR']['lower'] == float('-inf')
	assert pt2.base_node_dict['FNR']['upper'] == float('inf')
	assert pt2.base_node_dict['FNR']['bound_computed'] == False

