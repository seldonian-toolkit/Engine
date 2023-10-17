""" Module containing global variables used 
during the construction of parse trees 

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
import ast

op_mapper = {
    ast.Sub: "sub",
    ast.Add: "add",
    ast.Mult: "mult",
    ast.Div: "div",
    ast.Pow: "pow",
}

not_supported_op_mapper = {
    ast.Mod: "mod",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitAnd: "&",
    ast.FloorDiv: "//",
    ast.UAdd: "+",
    ast.Not: "not",
    ast.Invert: "~",
}

bounds_required_dict = {
    "add": {
        "lower": [1, 0, 1, 0],
        "upper": [0, 1, 0, 1],
    },
    "sub": {
        "lower": [1, 0, 0, 1],
        "upper": [0, 1, 1, 0],
    },
    "mult": {
        "lower": [1, 1, 1, 1],
        "upper": [1, 1, 1, 1],
    },
    "div": {
        "lower": [1, 1, 1, 1],
        "upper": [1, 1, 1, 1],
    },
    "pow": {
        "lower": [1, 1, 1, 1],
        "upper": [1, 1, 1, 1],
    },
    "min": {
        "lower": [1, 0, 1, 0],
        "upper": [0, 1, 0, 1],
    },
    "max": {
        "lower": [1, 0, 1, 0],
        "upper": [0, 1, 0, 1],
    },
    "abs": {
        "lower": [1, 1],
        "upper": [1, 1],
    },
    "exp": {
        "lower": [1, 0],
        "upper": [0, 1],
    },
    "log": {
        "lower": [1, 0],
        "upper": [0, 1],
    },
}
