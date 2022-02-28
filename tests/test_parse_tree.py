from src.parse_tree import *
import pytest

### Utilities for testing 
def stump(operator_type,left_bounds,right_bounds):
    # A parse tree with a root node and left and right children only
    root = InternalNode(operator_type)
    root.left = BaseNode('a')
    root.right = BaseNode('b')
    pt = ParseTree(delta=0.05)
    pt.root = root
    pt.root.left.lower  = left_bounds[0]
    pt.root.left.upper  = left_bounds[1]
    pt.root.right.lower = right_bounds[0]
    pt.root.right.upper = right_bounds[1]
    pt.n_nodes = 3
    pt.n_base_nodes = 2
    pt.base_node_dict = {
        'a':{
            'computed':False,
            'lower':float("-inf"),
            'upper':float("inf")
            },
        'b':{
            'computed':False,
            'lower':float("-inf"),
            'upper':float("inf")
            },
    }
    return pt

def edge(operator_type,left_bounds):
    # A parse tree with a single edge
    assert operator_type in ['abs','exp']
    root = InternalNode(operator_type)
    root.left = BaseNode('a')
    pt = ParseTree(delta=0.05)
    pt.root = root
    pt.root.left.lower  = left_bounds[0]
    pt.root.left.upper  = left_bounds[1]
    pt.n_nodes = 2
    pt.n_base_nodes = 1
    pt.base_node_dict = {
        'a':{
            'computed':False,
            'lower':float("-inf"),
            'upper':float("inf")
            },
    }
    return pt

two_interval_options = [
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
@pytest.mark.parametrize('interval_index',range(len(two_interval_options)))
def test_add_bounds(interval_index):
    ### Addition ###

    a,b=two_interval_options[interval_index]
    answer = answer_dict['add'][interval_index]
    pt = stump('add',a,b)
    pt.propagate_bounds(bound_method='manual')
    assert pt.root.lower == answer[0]
    assert pt.root.upper == answer[1]
    assert pt.base_node_dict['a']['computed'] == True
    assert pt.base_node_dict['b']['computed'] == True

@pytest.mark.parametrize('interval_index',range(len(two_interval_options)))
def test_subtract_bounds(interval_index):
    ### Subtraction ###

    a,b=two_interval_options[interval_index]
    answer = answer_dict['sub'][interval_index]
    pt = stump('sub',a,b)
    pt.propagate_bounds(bound_method='manual')
    # Use approx due to floating point imprecision
    assert pt.root.lower == pytest.approx(answer[0])
    assert pt.root.upper == pytest.approx(answer[1])
    assert pt.base_node_dict['a']['computed'] == True
    assert pt.base_node_dict['b']['computed'] == True

@pytest.mark.parametrize('interval_index',range(len(two_interval_options)))
def test_multiply_bounds(interval_index):
    ### Multiplication ###

    a,b=two_interval_options[interval_index]
    answer = answer_dict['mult'][interval_index]
    pt = stump('mult',a,b)
    pt.propagate_bounds(bound_method='manual')
    # Use approx due to floating point imprecision
    assert pt.root.lower == pytest.approx(answer[0])
    assert pt.root.upper == pytest.approx(answer[1])
    assert pt.base_node_dict['a']['computed'] == True
    assert pt.base_node_dict['b']['computed'] == True

@pytest.mark.parametrize('interval_index',range(len(two_interval_options)))
def test_divide_bounds(interval_index):
    ### Division ###

    a,b=two_interval_options[interval_index]
    answer = answer_dict['div'][interval_index]
    pt = stump('div',a,b)
    pt.propagate_bounds(bound_method='manual')
    # Use approx due to floating point imprecision
    assert pt.root.lower == pytest.approx(answer[0])
    assert pt.root.upper == pytest.approx(answer[1])
    assert pt.base_node_dict['a']['computed'] == True
    assert pt.base_node_dict['b']['computed'] == True

@pytest.mark.parametrize('interval_index',range(len(single_interval_options)))
def test_abs_bounds(interval_index):
    ### Absolute value ###

    a=single_interval_options[interval_index]
    answer = answer_dict['abs'][interval_index]
    pt = edge('abs',a)
    pt.propagate_bounds(bound_method='manual')
    # Use approx due to floating point imprecision
    assert pt.root.lower == pytest.approx(answer[0])
    assert pt.root.upper == pytest.approx(answer[1])
    assert pt.base_node_dict['a']['computed'] == True

def test_parse_tree_from_simple_string():

    constraint_str = 'x - (y + b)*4'
    delta = 0.05
    pt = ParseTree(delta)
    pt.create_from_ast(constraint_str)
    assert pt.n_nodes == 7
    assert pt.n_base_nodes == 3
    assert len(pt.base_node_dict) == 3
    assert isinstance(pt.root,InternalNode)
    assert pt.root.name == 'sub'
 
def test_parse_tree_with_special_base_variables():

    constraint_str = 'abs((Mean_Error|M) - (Mean_Error|F)) - 0.1'
    delta = 0.05
    pt = ParseTree(delta)
    pt.create_from_ast(constraint_str)
    assert pt.n_nodes == 6
    assert pt.n_base_nodes == 2
    assert len(pt.base_node_dict) == 2  
    assert isinstance(pt.root,InternalNode)
    assert pt.root.name == 'sub'  
    assert pt.root.left.name == 'abs'
    assert pt.root.right.value == 0.1

def test_deltas_assigned_equally():
    constraint_str = 'abs((Mean_Error|M) - (Mean_Error|F)) - 0.1'
    delta = 0.05 

    pt = ParseTree(delta)
    pt.create_from_ast(constraint_str)
    pt.assign_deltas(weight_method='equal')
    assert pt.n_nodes == 6
    assert pt.n_base_nodes == 2
    assert len(pt.base_node_dict) == 2  
    assert isinstance(pt.root,InternalNode)
    assert pt.root.name == 'sub'  
    assert pt.root.left.left.left.delta == delta/pt.n_base_nodes
    assert pt.root.left.left.right.delta == delta/pt.n_base_nodes

