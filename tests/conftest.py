import numpy as np
import pytest

from seldonian.parse_tree import *


@pytest.fixture
def stump():
    def stump_function(operator_type,left_bounds,right_bounds):
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
                'upper':float("inf"),
                'data':None,
                'datasize':0
                },
            'b':{
                'computed':False,
                'lower':float("-inf"),
                'upper':float("inf"),
                'data':None,
                'datasize':0
                },
        }
        return pt
    return stump_function

@pytest.fixture
def edge():
    def edge_function(operator_type,left_bounds):
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
                'upper':float("inf"),
                'data':None,
                'datasize':0
                },
        }
        return pt
    return edge_function

@pytest.fixture
def generate_data():
    def generate_data_function(numPoints,loc_X=0.0,loc_Y=0.0,sigma_X=1.0,sigma_Y=1.0):
        X =     np.random.normal(loc_X, sigma_X, numPoints) # Sample x from a standard normal distribution
        Y = X + np.random.normal(loc_Y, sigma_Y, numPoints) # Set y to be x, plus noise from a standard normal distribution
        return (X,Y)
    
    return generate_data_function

