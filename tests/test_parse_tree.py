from seldonian.parse_tree.parse_tree import *
from seldonian.dataset import (DataSetLoader,
	SupervisedDataSet)
from seldonian.safety_test.safety_test import SafetyTest
from seldonian.models.model import LinearRegressionModel
import pytest
import time

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
	pt.propagate_bounds(bound_method='manual')
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
	pt.propagate_bounds(bound_method='manual')
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
	pt.propagate_bounds(bound_method='manual')
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
	pt.propagate_bounds(bound_method='manual')
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
	print(a,b)
	pt = stump('pow',a,b)
	if a[0] < 0:
		with pytest.warns(UserWarning,match=warning_msg):
			with pytest.raises(ArithmeticError) as excinfo:
				pt.propagate_bounds(bound_method='manual')
		
		assert "Cannot compute interval" in str(excinfo.value)
		assert "because first argument contains negatives" in str(excinfo.value)

	elif 0 in a and (b[0]<0 or b[1]<1):
		with pytest.warns(UserWarning,match=warning_msg):
			with pytest.raises(ZeroDivisionError) as excinfo:
				pt.propagate_bounds(bound_method='manual')
		
		assert "0.0 cannot be raised to a negative power" in str(excinfo.value)
	else:
		answer = answer_dict['pow'][interval_index]
		print(answer)
		with pytest.warns(UserWarning,match=warning_msg):
			pt.propagate_bounds(bound_method='manual')
		
		# Use approx due to floating point imprecision
		assert pt.root.lower == pytest.approx(answer[0])
		assert pt.root.upper == pytest.approx(answer[1])
		assert pt.base_node_dict['a']['bound_computed'] == True
		assert pt.base_node_dict['b']['bound_computed'] == True

@pytest.mark.parametrize('interval_index',range(len(two_interval_options)))
def test_min_bounds(interval_index,stump):
	### min ###

	a,b=two_interval_options[interval_index]
	print(a,b)
	pt = stump('min',a,b)
	
	answer = answer_dict['min'][interval_index]
	print(answer)
	pt.propagate_bounds(bound_method='manual')
	# Use approx due to floating point imprecision
	assert pt.root.lower == pytest.approx(answer[0])
	assert pt.root.upper == pytest.approx(answer[1])
	assert pt.base_node_dict['a']['bound_computed'] == True
	assert pt.base_node_dict['b']['bound_computed'] == True

@pytest.mark.parametrize('interval_index',range(len(two_interval_options)))
def test_max_bounds(interval_index,stump):
	### min ###

	a,b=two_interval_options[interval_index]
	print(a,b)
	pt = stump('max',a,b)
	
	answer = answer_dict['max'][interval_index]
	print(answer)
	pt.propagate_bounds(bound_method='manual')
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
	pt.propagate_bounds(bound_method='manual')
	# Use approx due to floating point imprecision
	assert pt.root.lower == pytest.approx(answer[0])
	assert pt.root.upper == pytest.approx(answer[1])
	assert pt.base_node_dict['a']['bound_computed'] == True

########################
### Parse tree tests ###
########################

def test_parse_tree_from_simple_string():
	constraint_str = 'FPR - (FNR + PR)*4'
	delta = 0.05
	pt = ParseTree(delta)
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

def test_measure_functions_recognized():
	delta = 0.05

	constraint_str = 'Mean_Squared_Error - 2.0'
	pt = ParseTree(delta)
	pt.create_from_ast(constraint_str)
	assert pt.root.left.measure_function_name == 'Mean_Squared_Error'
	
	constraint_str = '(Mean_Error|[M]) - 2.0'

	pt = ParseTree(delta)
	pt.create_from_ast(constraint_str)
	assert pt.root.left.measure_function_name == 'Mean_Error'

	constraint_str = '(FPR|[A,B]) - 2.0'

	pt = ParseTree(delta)
	pt.create_from_ast(constraint_str)
	assert pt.root.left.measure_function_name == 'FPR'

	# Test that a non-measure base node 
	# is not recognized as measure
	constraint_str = 'X - 2.0'

	pt = ParseTree(delta)
	with pytest.raises(NotImplementedError) as excinfo:
		pt.create_from_ast(constraint_str)
	
	error_str = ("Error parsing your expression."
			 " A variable name was used which we do not recognize: X")
	assert str(excinfo.value) == error_str

	# Test that a non-measure base node 
	# is not recognized as measure
	constraint_str = '(X | [A]) - 2.0'

	pt = ParseTree(delta)
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
		pt = ParseTree(delta)
		with pytest.raises(RuntimeError) as excinfo:
			pt.create_from_ast(constraint_str)
		
		assert str(excinfo.value) == error_str
 
def test_custom_base_node():
	constraint_str = 'MED_MF - 0.1'
	delta = 0.05 

	pt = ParseTree(delta)
	pt.create_from_ast(constraint_str)
	assert isinstance(pt.root.left,BaseNode)
	assert pt.n_base_nodes == 1
	assert len(pt.base_node_dict) == 1

def test_unary_op():
	delta = 0.05 

	constraint_str = '-10+abs(Mean_Error)'
	pt = ParseTree(delta)
	pt.create_from_ast(constraint_str)
	assert pt.root.name == 'add'
	assert pt.root.left.value == -10
	assert pt.root.left.name == '-10'
	assert pt.root.right.name == 'abs'
	assert pt.root.right.left.name == 'Mean_Error'
	assert pt.n_nodes == 4
	assert pt.n_base_nodes == 1

	constraint_str = '-MED_MF'
	pt = ParseTree(delta)
	pt.create_from_ast(constraint_str)
	assert pt.root.name == 'mult'
	assert pt.root.left.value == -1
	assert pt.root.right.name == 'MED_MF'
	assert pt.n_nodes == 3


	constraint_str = '-abs(Mean_Error) - 10'
	pt = ParseTree(delta)
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
	pt = ParseTree(delta)
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
	pt = ParseTree(delta)
	with pytest.raises(NotImplementedError) as excinfo:
		pt.create_from_ast(constraint_str)
	error_str = ("Error parsing your expression."
		 " An operator was used which we do not support: ^")
	assert str(excinfo.value) == error_str

	constraint_str = 'FPR<<4'
	delta = 0.05
	pt = ParseTree(delta)
	with pytest.raises(NotImplementedError) as excinfo:
		pt.create_from_ast(constraint_str)
	error_str = ("Error parsing your expression."
		 " An operator was used which we do not support: <<")
	assert str(excinfo.value) == error_str

	constraint_str = 'FPR>>4'
	delta = 0.05
	pt = ParseTree(delta)
	with pytest.raises(NotImplementedError) as excinfo:
		pt.create_from_ast(constraint_str)
	error_str = ("Error parsing your expression."
		 " An operator was used which we do not support: >>")
	assert str(excinfo.value) == error_str

	constraint_str = 'FPR & FNR'
	delta = 0.05
	pt = ParseTree(delta)
	with pytest.raises(NotImplementedError) as excinfo:
		pt.create_from_ast(constraint_str)
	error_str = ("Error parsing your expression."
		 " An operator was used which we do not support: &")
	assert str(excinfo.value) == error_str

	constraint_str = 'FPR//4'
	delta = 0.05
	pt = ParseTree(delta)
	with pytest.raises(NotImplementedError) as excinfo:
		pt.create_from_ast(constraint_str)
	error_str = ("Error parsing your expression."
		 " An operator was used which we do not support: //")
	assert str(excinfo.value) == error_str
 
def test_single_conditional_columns_assigned():

	constraint_str = 'abs(Mean_Error|[X]) - 0.1'
	delta = 0.05
	pt = ParseTree(delta)
	pt.create_from_ast(constraint_str)
	assert pt.n_nodes == 4
	assert pt.n_base_nodes == 1
	assert len(pt.base_node_dict) == 1  
	assert pt.root.left.left.conditional_columns == ['X']

def test_multiple_conditional_columns_assigned():

	constraint_str = 'abs(Mean_Error|[X,Y,Z]) - 0.1'
	delta = 0.05
	pt = ParseTree(delta)
	pt.create_from_ast(constraint_str)
	assert pt.n_nodes == 4
	assert pt.n_base_nodes == 1
	assert len(pt.base_node_dict) == 1  
	assert pt.root.left.left.conditional_columns == ['X','Y','Z']

def test_deltas_assigned_equally():
	constraint_str = 'abs((Mean_Error|[M]) - (Mean_Error|[F])) - 0.1'
	delta = 0.05 

	pt = ParseTree(delta)
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

	pt = ParseTree(delta)
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
	pt = ParseTree(delta)
	pt.create_from_ast(constraint_str)
	# Before bounds assigned both should be True
	assert pt.root.will_lower_bound == True
	assert pt.root.will_upper_bound == True
	pt.assign_bounds_needed()
	# But after, we should find that only upper is needed
	assert pt.n_nodes == 1
	assert pt.n_base_nodes == 1  
	assert isinstance(pt.root,BaseNode)
	# print(pt.root.will_lower_bound)
	assert pt.root.will_lower_bound == False
	assert pt.root.will_upper_bound == True

	constraint_str = '(Mean_Error | [M]) - 0.1'
	pt = ParseTree(delta)
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
	pt = ParseTree(delta)
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
	pt = ParseTree(delta)
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
	pt = ParseTree(delta)
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
	pt = ParseTree(delta)
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
	pt = ParseTree(delta)
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

	pt = ParseTree(delta)
	pt.create_from_ast(constraint_str)
	assert pt.n_base_nodes == 2 
	assert len(pt.base_node_dict) == 1 
	assert pt.base_node_dict['FPR']['bound_computed'] == False
	pt.propagate_bounds(bound_method='random')
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
	dataset = SupervisedDataSet(df,meta_information=['feature1','label'],
		regime='supervised',label_column='label',
		include_sensitive_columns=False,
		include_intercept_term=True)
	
	constraint_str = 'Mean_Squared_Error - 2.0'
	delta = 0.05 

	pt = ParseTree(delta)
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
	pt.propagate_bounds(theta=theta,dataset=dataset,
		model=model_instance,
		branch='safety_test',
		bound_method='ttest',
		regime='supervised')
	assert pt.root.lower == float('-inf') # not bound_computed 
	assert pt.root.upper == pytest.approx(-0.995242)

def test_evaluate_constraint(generate_data):
	# Evaluate constraint mean, not the bound
	np.random.seed(0)
	numPoints=1000

	model_instance = LinearRegressionModel()
	X,Y = generate_data(
		numPoints,loc_X=0.0,loc_Y=0.0,sigma_X=1.0,sigma_Y=1.0)

	rows = np.hstack([np.expand_dims(X,axis=1),np.expand_dims(Y,axis=1)])
	
	df = pd.DataFrame(rows,columns=['feature1','label'])
	
	dataset = SupervisedDataSet(df,meta_information=['feature1','label'],
		regime='supervised',label_column='label',
		include_sensitive_columns=False,
		include_intercept_term=True)
	
	constraint_str = 'Mean_Squared_Error - 2.0'
	delta = 0.05 

	pt = ParseTree(delta)
	pt.create_from_ast(constraint_str)

	pt.assign_deltas(weight_method='equal')
	pt.assign_bounds_needed()
	assert pt.n_nodes == 3
	assert pt.n_base_nodes == 1
	assert len(pt.base_node_dict) == 1
	
	theta = np.array([0,1])
	pt.evaluate_constraint(theta=theta,dataset=dataset,
		model=model_instance,regime='supervised',
		branch='safety_test')
	print(pt.root.value)
	assert pt.root.value == pytest.approx(-1.06248)

def test_reset_parse_tree():
	
	constraint_str = '(FPR + FNR) - 0.5'
	delta = 0.05 

	pt = ParseTree(delta)
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

	# propagate bounds
	pt.propagate_bounds(bound_method='random')
	assert len(pt.base_node_dict) == 2
	assert pt.base_node_dict['FPR']['bound_computed'] == True
	assert pt.base_node_dict['FNR']['bound_computed'] == True
	assert pt.base_node_dict['FPR']['lower'] >= 0
	assert pt.base_node_dict['FPR']['upper'] > 0
	assert pt.base_node_dict['FNR']['lower'] >= 0
	assert pt.base_node_dict['FNR']['upper'] > 0

	# reset the node dict 
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
	csv_file = 'static/datasets/GPA/gpa_regression_dataset.csv'
	columns = ["M","F","SAT_Physics",
		   "SAT_Biology","SAT_History",
		   "SAT_Second_Language","SAT_Geography",
		   "SAT_Literature","SAT_Portuguese_and_Essay",
		   "SAT_Math","SAT_Chemistry","GPA"]
		   
	loader = DataSetLoader(column_names=columns,
		sensitive_column_names=['M','F'],
		regime='supervised',label_column='GPA',
		include_sensitive_columns=False,
		include_intercept_term=True)
	dataset = loader.from_csv(csv_file)

	model_instance = LinearRegressionModel()

	constraint_str = 'abs(Mean_Error|[M]) - 0.1'
	delta = 0.05
	pt = ParseTree(delta)
	pt.create_from_ast(constraint_str)
	pt.assign_deltas(weight_method='equal')

	# propagate the bounds with example theta value
	# theta = np.hstack([np.array([0.0,0.0]),np.random.uniform(-0.05,0.05,10)])
	theta = np.random.uniform(-0.05,0.05,10)
	pt.propagate_bounds(theta=theta,dataset=dataset,
		model=model_instance,branch='safety_test',
		bound_method='ttest',
		regime='supervised')
	assert pt.root.lower == pytest.approx(61.9001779655)
	assert pt.root.upper == pytest.approx(62.1362236720)
	print(pt.base_node_dict.keys())
	assert len(pt.base_node_dict["Mean_Error | [M]"]['data_dict']['features']) == 22335
	pt.reset_base_node_dict()
	