import ast
import warnings

import graphviz
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from seldonian.warnings.custom_warnings import *
from seldonian.nodes import *
from seldonian.constraints.constraints import *


"""
Define a map containing which child bounds are required for each operator

If an operator has two children, A and B then
arrays are boolean of length 4, like: 
[need_A_lower,need_A_upper,need_B_lower,need_B_upper]

If an operator has one child, A, then
arrays are boolean:
[need_A_lower, need_A_upper]
"""
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


class ParseTree(object):
	""" 
	Class to represent a parse tree for a single behavioral constraint

	Attributes
	----------
	root : nodes.Node class instance
		Root node which contains the whole tree 
		via left and right child attributes.
		Gets assigned when tree is built
	constraint_str: str
		The string expression for the behavioral
		constraint
	delta: float
		Confidence level. Specifies the maximum probability 
		that the algorithm can return a solution violat the
		behavioral constraint.
	n_nodes: int
		Total number of nodes in the parse tree
	n_base_nodes: int
		Number of base variable nodes in the parse tree.
		Does not include constants. If a base variable,
		such as PR | [M] appears more than once in the 
		constraint_str each appearance contributes 
		to n_base_nodes
	base_node_dict: dict
		Keeps track of unique base variable nodes,
		their confidence bounds and whether 
		the bounds have been calculated
		for a given base node already.
		Helpful for handling case where we have 
		duplicate base nodes 
	node_fontsize: int
		Fontsize used for graphviz visualizations

	Methods
	-------
	create_from_ast(s)
		Create the node structure of the tree
		given a mathematical string expression
		for the behavioral constraint, s

	_ast_tree_helper(ast_node)
		Helper function for create_from_ast()

	_ast2pt_node(ast_node)
		Mapper between python's ast library's
		node objects to this library's Node objects

	assign_deltas(weight_method)
		Assign the delta values to the base nodes in the tree

	_assign_deltas_helper(node,weight_method)
		Helper function for assign_deltas()
	
	assign_bounds_needed()
		Assign whether lower, upper or both bounds
		need to be calculated for each node.

	_assign_bounds_helper()
		Helper function for assign_bounds_needed

	propagate_bounds()
		Traverse the parse tree, calculate confidence
		bounds on base nodes and 
		then propagate bounds using propagation logic

	_propagator_helper(node)
		Helper function for propagate_bounds()

	evaluate_constraint()
		Get the mean value of the constraint,
		given a dataset and model

	_evaluator_helper(node)
		Helper function for evaluate_constraint()

	_propagate_value(node)
		Given an internal node, calculate the 
		propagated node value from its children 
		using the node's operator type

	_protect_nan(bound,bound_type)
		Handle nan as negative infinity if in lower bound
		and postitive infinity if in upper bound 

	_propagate(node)
		Given an internal node, calculate 
		the propagated confidence interval
		from its children using the 
		node's operator type

	_add(a,b)
		Add intervals a and b

	_sub(a,b)
		Subtract intervals a and b

	_mult(a,b)
		Multiply intervals a and b

	_div(a,b)
		Divide intervals a and b  

	_pow(a,b)
		Raise interval a to the power of interval b 
		(experimental feature)
	
	_min(a,b)
		Get the minimum interval from intervals a and b

	_max(a,b)
		Get the maximum interval from intervals a and b

	_abs(a)
		Take the absolute value of interval a 

	_exp(a)
		Calculate e raised to the interval a.

	reset_base_node_dict(reset_data=False)

	make_viz(title)
		Make a graphviz graph object of 
		the parse tree and give it a title

	make_viz_helper(root,graph)
		Helper function for make_viz()

	"""
	def __init__(self,delta):
		self.root = None 
		self.constraint_str = ''
		self.delta = delta
		self.n_nodes = 0
		self.n_base_nodes = 0
		self.base_node_dict = {} 
		self.node_fontsize = 12

	def create_from_ast(self,s):
		""" 
		Create the node structure of the tree
		given a mathematical string expression, s

		Parameters
		----------
		s : str
			mathematical expression written in Python syntax
			from which we build the parse tree
		"""
		self.constraint_str = s
		self.node_index = 0

		tree = ast.parse(s)
		# makes sure this is a single expression
		assert len(tree.body) == 1 

		expr = tree.body[0]
		root = expr.value

		# Recursively build the tree
		self.root = self._ast_tree_helper(root)

	def _ast_tree_helper(self,ast_node):
		""" 
		From a given node in the ast tree,
		make a node in our tree and recurse
		to children of this node.

		Parameters
		----------
		node : ast.AST node class instance 
			
		"""
		# base case
		if ast_node is None:
			return None

		is_parent = False
		
		# handle unary operator like "-var" 
		if isinstance(ast_node,ast.UnaryOp):

			# Only handle unary "-", reject rest	
			if ast_node.op.__class__ != ast.USub:
				op = not_supported_op_mapper[ast_node.op.__class__]
				raise NotImplementedError("Error parsing your expression."
					" A unary operator was used which we do not support: "
					f"{op}")
			
			# If operand is a constant, make a ConstantNode
			# with a negative value
			if isinstance(ast_node.operand,ast.Constant):
				node_value = -ast_node.operand.value
				node_name = str(-ast_node.operand.value)
				is_leaf = True
				new_node = ConstantNode(node_name,node_value)
			else:
				# Make three nodes, -1, * and whatever the operand is
				new_node_parent = InternalNode('mult')
				self.n_nodes += 1
				new_node_parent.index = self.node_index
				self.node_index +=1

				new_node_parent.left = ConstantNode('-1',-1.0)
				self.n_nodes += 1
				new_node_parent.left.index = self.node_index
				self.node_index +=1
				
				new_node, is_leaf =  self._ast2pt_node(ast_node.operand)
				new_node_parent.right = new_node
				new_node_parent.right.index = self.node_index
				is_parent = True
				ast_node = ast_node.operand
			
		else: 
			new_node,is_leaf = self._ast2pt_node(ast_node)


		if isinstance(new_node,BaseNode):
			self.n_base_nodes += 1

			# strip out conditional columns and parentheses
			# to get the measure function name
			# does not fail if none are present
			node_name_isolated = new_node.name.split(
				"|")[0].strip().strip('(').strip()
			if node_name_isolated in measure_functions:
				new_node.measure_function_name = node_name_isolated		

			# if node with this name not already in self.base_node_dict
			# then make a new entry 
			if new_node.name not in self.base_node_dict: 
				self.base_node_dict[new_node.name] = {
					'bound_computed':False,
					'value_computed':False,
					'lower':float('-inf'),
					'upper':float('inf'),
					'data_dict':None,
					'datasize':0
				}

		self.n_nodes += 1
		new_node.index = self.node_index
		self.node_index +=1

		# If node is a leaf node, don't check for children
		if is_leaf:
			if is_parent:
				return new_node_parent
			return new_node
		# otherwise we are at an internal node
		# and need to recurse
		if hasattr(ast_node,'left'):
			new_node.left = self._ast_tree_helper(ast_node.left)
		if hasattr(ast_node,'right'):
			new_node.right = self._ast_tree_helper(ast_node.right)
		
		# Handle functions like min(), abs(), etc...
		if hasattr(ast_node,'args') and ast_node.func.id not in measure_functions:
			if len(ast_node.args) == 0 or len(ast_node.args) > 2: 
				readable_args = [x.id for x in ast_node.args]
				raise NotImplementedError(
					"Please check the syntax of the function: "
				   f" {new_node.name}(), with arguments: {readable_args}")
			for ii,arg in enumerate(ast_node.args):
				if ii == 0:
					new_node.left = self._ast_tree_helper(arg)
				if ii == 1:
					new_node.right = self._ast_tree_helper(arg)

		if is_parent:
			return new_node_parent
		return new_node

	def _ast2pt_node(self,ast_node):
		""" 
		Mapper to convert ast.AST node objects
		to our Node objects

		Parameters
		----------
		ast_node : ast.AST node class instance
		"""
		is_leaf = False
		kwargs = {}
		if isinstance(ast_node,ast.Tuple):
			raise RuntimeError(
				"Error parsing your expression."
				" The issue is most likely due to"
				" missing/mismatched parentheses or square brackets"
				" in a conditional expression involving '|'.")
		
		if isinstance(ast_node,ast.BinOp):
			# +,-,*,/,**,| operators
			if ast_node.op.__class__ == ast.BitOr:
				# BitOr is the "|" operator, used to represent
				# a "A | B" -> "A given B"
				
				node_class = BaseNode

				try:
					conditional_columns = [str(x.id) for x in ast_node.right.elts]
					conditional_columns_liststr = '[' + ''.join(conditional_columns) + ']'
					left_id = ast_node.left.id
				except:
					raise RuntimeError(
						"Error parsing your expression."
						" The issue is most likely due to"
						" missing/mismatched parentheses or square brackets"
						" in a conditional expression involving '|'.")
				
				if left_id not in measure_functions:
					raise NotImplementedError("Error parsing your expression."
						" A variable name was used which we do not recognize: "
					   f"{ast_node.left.id}")
				node_name = ' | '.join(
					[ast_node.left.id,conditional_columns_liststr])

				is_leaf = True
				return node_class(node_name,
					conditional_columns=conditional_columns),is_leaf
			else:
				node_class = InternalNode
				try:
					node_name = op_mapper[ast_node.op.__class__]
				except KeyError:
					op = not_supported_op_mapper[ast_node.op.__class__]
					raise NotImplementedError("Error parsing your expression."
						" An operator was used which we do not support: "
					   f"{op}")
				return node_class(node_name),is_leaf

		elif isinstance(ast_node,ast.Name):
			# named quantity like "e", "Mean_Squared_Error"
			# Custom base nodes will be caught here too
			# If variable name is "e" then make it a constant, not a base variable
			if ast_node.id == 'e':
				node_name = 'e'
				node_class = ConstantNode
				node_value = np.e
				is_leaf = True
				return node_class(node_name,node_value),is_leaf
			else:	
				if ast_node.id in custom_base_node_dict:
					# A user-defined base node 
					node_class = custom_base_node_dict[ast_node.id]
					node_name = ast_node.id

				elif ast_node.id not in measure_functions:
					raise NotImplementedError("Error parsing your expression."
						" A variable name was used which we do not recognize: "
					   f"{ast_node.id}")
				else:
					# a measure function in our list
					node_class = BaseNode
					node_name = ast_node.id
				
				is_leaf = True
				return node_class(node_name),is_leaf

		elif isinstance(ast_node,ast.Constant):
			# A constant floating point or integer number
			node_class = ConstantNode
			node_value = ast_node.value
			node_name = str(node_value)
			is_leaf = True
			return node_class(node_name,node_value),is_leaf

		elif isinstance(ast_node,ast.Call):
			# a function call like abs(arg1), min(arg1,arg2)
			node_class = InternalNode
			node_name = ast_node.func.id

		return node_class(node_name),is_leaf

	def assign_deltas(self,weight_method='equal',
		**kwargs):
		""" 
		Assign the delta values to the base nodes in the tree.

		Parameters
		----------
		weight_method : str
			How you want to assign the deltas to the base nodes
			'equal' : split up delta equally among base nodes 
		"""
		assert self.n_base_nodes > 0, (
			"Number of base nodes must be > 0."
			" Make sure to build the tree before assigning deltas.")
		self._assign_deltas_helper(self.root,weight_method,**kwargs)
		
	def _assign_deltas_helper(self,node,weight_method,**kwargs):
		""" 
		Helper function to traverse the parse tree 
		and assign delta values to base nodes.
		--TODO-- 
		Currently uses preorder, but there is likely
		a faster way to do this because if you get 
		to a base node, you know none 
		of its parents are possible base nodes

		Parameters
		----------
		node : nodes.Node class instance
		weight_method : str
			How you want to assign the deltas to the base nodes
				'equal' : split up delta equally among base nodes 
		"""
		
		if not node:
			return

		if isinstance(node,BaseNode): # captures all child classes of BaseNode as well
			if weight_method == 'equal':
				node.delta = self.delta/len(self.base_node_dict)

		self._assign_deltas_helper(node.left,weight_method)
		self._assign_deltas_helper(node.right,weight_method)
		return

	def assign_bounds_needed(self,**kwargs):
		""" 
		BFS through the tree and decide which bounds
		are required to compute on each child node.
		Eventually we get to base nodes 
		There are cases where it is not always 
		necessary to compute both lower and upper 
		bounds because at the end all we care about
		is the upper bound of the root node. 

		"""
		assert self.n_nodes > 0, "Number of nodes must be > 0"
		# initialize needed bounds for root
		lower_needed = False
		upper_needed = True
		self._assign_bounds_helper(self.root,
			lower_needed,upper_needed,**kwargs)
		
	def _assign_bounds_helper(self,node,
		lower_needed,upper_needed,
		**kwargs):
		""" 
		Helper function to traverse the parse tree 
		and assign which bounds we need to calculate 
		on the base nodes.
		--TODO-- 
		Currently uses preorder, but there is likely
		a faster way to do this because if you get 
		to a base node, you know none 
		of its parents are possible base nodes

		Parameters
		----------
		node : nodes.Node class instance
		lower_needed : bool
			Whether lower bound needs to be calculated
		upper_needed : bool
			Whether upper bound needs to be calculated
		"""

		# if we go off the end return
		if not node:
			return
		node.will_lower_bound = lower_needed
		node.will_upper_bound = upper_needed
		
		# If we get to a base node or constant node, then return
		if isinstance(node,BaseNode) or isinstance(node,ConstantNode): 
			return

		if isinstance(node,InternalNode):
			# depending on operator type and current bounds 
			# needed in the parent, determine which bounds
			# need to be calculated on the child nodes
			
			bounds_dict = bounds_required_dict[node.name]

			two_children = True
			if len(bounds_dict['lower']) == 2:
				two_children = False

			if lower_needed and upper_needed:
				if two_children:
					(left_lower_needed,
					left_upper_needed,
					right_lower_needed,
					right_upper_needed) = np.logical_or(
						bounds_dict['lower'],
						bounds_dict['upper']
					)
				else:
					(left_lower_needed,
					left_upper_needed) = np.logical_or(
						bounds_dict['lower'],
						bounds_dict['upper']
					)

			elif lower_needed or upper_needed:
				# only one bound is needed
				if lower_needed:
					if two_children:	
						(left_lower_needed,
						left_upper_needed,
						right_lower_needed,
						right_upper_needed) = bounds_dict['lower']
					else:
						(left_lower_needed,
						left_upper_needed) = bounds_dict['lower']

				if upper_needed:
					if two_children:
						(left_lower_needed,
						left_upper_needed,
						right_lower_needed,
						right_upper_needed) = bounds_dict['upper']
					else:
						(left_lower_needed,
						left_upper_needed) = bounds_dict['upper']			
			else:
				raise RuntimeError("Need at least lower or upper bound")

			self._assign_bounds_helper(node.left,
				left_lower_needed,left_upper_needed)

			if two_children:
				self._assign_bounds_helper(node.right,
					right_lower_needed,right_upper_needed)
			return

	def propagate_bounds(self,
		**kwargs):
		""" 
		Postorder traverse (left, right, root)
		through the tree and calculate confidence
		bounds on base nodes,
		then propagate bounds using propagation logic

		"""
		if not self.root:
			return []

		self._propagator_helper(self.root,
			**kwargs)
	
	def _propagator_helper(self,node,
		**kwargs):
		""" 
		Helper function for traversing 
		through the tree and propagating confidence bounds

		Parameters
		----------
		node : nodes.Node class instance	
		"""

		# if we hit a constant node or run past the end of the tree
		# return because we don't need to calculate bounds
		if not node or isinstance(node,ConstantNode):
			return 

		# if we hit a BaseNode,
		# then calculate confidence bounds and return 
		if isinstance(node,BaseNode):
			# Check if bound has already been calculated for this node name
			# If so, use precalculated bound
			if self.base_node_dict[node.name]['bound_computed'] == True:
				node.lower = self.base_node_dict[node.name]['lower']
				node.upper = self.base_node_dict[node.name]['upper'] 
				return
			else:
				if 'dataset' in kwargs:
					# Check if data has already been prepared
					# for this node name. If so, use precalculated data
					if self.base_node_dict[node.name]['data_dict']!=None:
						data_dict = self.base_node_dict[node.name]['data_dict']
						datasize = self.base_node_dict[node.name]['datasize']
					else:
						data_dict,datasize = node.calculate_data_forbound(
							**kwargs)
						self.base_node_dict[node.name]['data_dict'] = data_dict
						self.base_node_dict[node.name]['datasize'] = datasize

					kwargs['data_dict'] = data_dict
					kwargs['datasize'] = datasize

				bound_result = node.calculate_bounds(
					**kwargs)
				self.base_node_dict[node.name]['bound_computed'] = True
				
				if node.will_lower_bound:
					node.lower = bound_result['lower']
					self.base_node_dict[node.name]['lower'] = node.lower

				if node.will_upper_bound:
					node.upper = bound_result['upper']
					self.base_node_dict[node.name]['upper'] = node.upper
				
			return 
		
		# traverse to children first
		self._propagator_helper(node.left,
			**kwargs)
		self._propagator_helper(node.right,
			**kwargs)
		
		# Here we must be at an internal node and therefore need to propagate
		node.lower,node.upper = self._propagate(node)
	
	def evaluate_constraint(self,
		**kwargs):
		""" 
		Evaluate the constraint itself (not bounds_)
		Postorder traverse (left, right, root)
		through the tree and calculate the values
		of the base nodes 
		then propagate bounds using propagation logic
		"""
		if not self.root:
			return []

		self._evaluator_helper(self.root,
			**kwargs)

	def _evaluator_helper(self,node,
		**kwargs):
		""" 
		Helper function for traversing 
		through the tree to evaluate the constraint

		Parameters
		----------
		node : nodes.Node class instance
		"""

		# if we hit a constant node or run past the end of the tree
		# return because we don't need to calculate anything
		if not node or isinstance(node,ConstantNode):
			return 

		# if we hit a BaseNode,
		# then calculate the value and return 
		if isinstance(node,BaseNode):
			# Check if value has already been calculated for this node name
			# If so, use precalculated value
			if self.base_node_dict[node.name]['value_computed'] == True:
				node.value = self.base_node_dict[node.name]['value']
				return
			else:
				if 'dataset' in kwargs:
					# Check if data has already been prepared
					# for this node name. If so, use precalculated data
					if self.base_node_dict[node.name]['data_dict']!=None:
						data_dict = self.base_node_dict[node.name]['data_dict']
						datasize = self.base_node_dict[node.name]['datasize']
					else:
						data_dict,datasize = node.calculate_data_forbound(
							**kwargs)
						self.base_node_dict[node.name]['data_dict'] = data_dict
						self.base_node_dict[node.name]['datasize'] = datasize

					kwargs['data_dict'] = data_dict
					kwargs['datasize'] = datasize
				value = node.calculate_value(
					**kwargs)
				node.value = value
				self.base_node_dict[node.name]['value_computed'] = True	
				self.base_node_dict[node.name]['value'] = node.value
				
			return 
		
		# traverse to children first
		self._evaluator_helper(node.left,
			**kwargs)
		self._evaluator_helper(node.right,
			**kwargs)
		
		# Here we must be at an internal node and therefore need to propagate
		node.value = self._propagate_value(node)
	
	def _propagate_value(self,node):
		"""
		Helper function for propagating values

		Parameters
		----------
		node : nodes.Node class instance
		"""
		a = node.left.value
		if node.right:
			b = node.right.value

		if node.name == 'add':	
			return a+b
			
		if node.name == 'sub':
			return a-b
			
		if node.name == 'mult':
			return a*b

		if node.name == 'div':
			return a/b 
		
		if node.name == 'pow':
			warning_msg = ("Warning: Power operation "
				"is an experimental feature. Use with caution.")
			return pow(a,b)

		if node.name == 'min':
			return min(a,b)

		if node.name == 'max':
			return max(a,b)

		if node.name == 'abs':
			# takes one node
			return abs(a)
		
		if node.name == 'exp':
			# takes one node
			return np.exp(a)

		else:
			raise NotImplementedError("Encountered an operation we do not yet support", node.name)

	def _protect_nan(self,bound,bound_type):
		""" 
		Handle nan as negative infinity if in lower bound
		and postitive infinity if in upper bound 

		Parameters
		----------
		bound : float
			The value of the upper or lower bound 
		bound_type : str
			'lower' or 'upper'
		"""
		if np.isnan(bound):
			if bound_type == 'lower':
				return float('-inf')
			if bound_type == 'upper':
				return float('inf')
		else:
			return bound

	def _propagate(self,node):
		"""
		Helper function for propagating confidence bounds

		Parameters
		----------
		node : nodes.Node class instance
		"""
		if node.name == 'add':
			a = (node.left.lower,node.left.upper)
			b = (node.right.lower,node.right.upper)
			return self._add(a,b)
			
		if node.name == 'sub':
			a = (node.left.lower,node.left.upper)
			b = (node.right.lower,node.right.upper)
			return self._sub(a,b)
			
		if node.name == 'mult':
			a = (node.left.lower,node.left.upper)
			b = (node.right.lower,node.right.upper)
			return self._mult(a,b)

		if node.name == 'div':
			a = (node.left.lower,node.left.upper)
			b = (node.right.lower,node.right.upper)
			return self._div(a,b) 
		
		if node.name == 'pow':
			warning_msg = ("Warning: Power operation "
				"is an experimental feature. Use with caution.")
			warnings.warn(warning_msg)
			a = (node.left.lower,node.left.upper)
			b = (node.right.lower,node.right.upper)
			return self._pow(a,b)

		if node.name == 'min':
			a = (node.left.lower,node.left.upper)
			b = (node.right.lower,node.right.upper)
			return self._min(a,b)

		if node.name == 'max':
			a = (node.left.lower,node.left.upper)
			b = (node.right.lower,node.right.upper)
			return self._max(a,b)

		if node.name == 'abs':
			# takes one node
			a = (node.left.lower,node.left.upper)
			return self._abs(a)
		
		if node.name == 'exp':
			# takes one node
			a = (node.left.lower,node.left.upper)
			return self._exp(a)

		else:
			raise NotImplementedError("Encountered an operation we do not yet support", node.name)
	
	def _add(self,a,b):
		"""
		Add two confidence intervals

		Parameters
		----------
		a : tuple
			Confidence interval like: (lower,upper)
		b : tuple
			Confidence interval like: (lower,upper)
		"""
		lower = self._protect_nan(
			a[0] + b[0],
			'lower')

		upper = self._protect_nan(
			a[1] + b[1],
			'upper')
		
		return (lower,upper)

	def _sub(self,a,b):
		"""
		Subract two confidence intervals

		Parameters
		----------
		a : tuple
			Confidence interval like: (lower,upper)
		b : tuple
			Confidence interval like: (lower,upper)
		"""
		lower = self._protect_nan(
				a[0] - b[1],
				'lower')
			
		upper = self._protect_nan(
			a[1] - b[0],
			'upper')

		return (lower,upper)

	def _mult(self,a,b):
		"""
		Multiply two confidence intervals

		Parameters
		----------
		a : tuple
			Confidence interval like: (lower,upper)
		b : tuple
			Confidence interval like: (lower,upper)
		"""        
		lower = self._protect_nan(
			min(a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]),
			'lower')
		
		upper = self._protect_nan(
			max(a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]),
			'upper')

		return (lower,upper)

	def _div(self,a,b):
		"""
		Divide two confidence intervals

		Parameters
		----------
		a : tuple
			Confidence interval like: (lower,upper)
		b : tuple
			Confidence interval like: (lower,upper)
		"""

		if b[0] < 0 < b[1]:
			# unbounded 
			lower = float('-inf')
			upper = float('inf')

		elif b[1] == 0:
			# reduces to multiplication of a*(-inf,1/b[0]]
			new_b = (float('-inf'),1/b[0])
			lower,upper = self._mult(a,new_b)

		elif b[0] == 0:
			# reduces to multiplication of a*(1/b[1],+inf)
			new_b = (1/b[1],float('inf'))
			lower,upper = self._mult(a,new_b)
		else:
			# b is either entirely negative or positive
			# reduces to multiplication of a*(1/b[1],1/b[0])
			new_b = (1/b[1],1/b[0])
			lower, upper = self._mult(a,new_b)

		return (lower,upper)

	def _pow(self,a,b):
		"""
		Get the confidence interval on 
		pow(a,b) where 
		b and b are both be intervals 

		Parameters
		----------
		a : tuple
			Confidence interval like: (lower,upper)
		b : tuple
			Confidence interval like: (lower,upper)
		"""

		# First, cases that are not allowed
		if a[0] < 0:
			raise ArithmeticError(
				f"Cannot compute interval: pow({a},{b}) because first argument contains negatives")
		if 0 in a and (b[0]<0 or b[1]<1):
			raise ZeroDivisionError("0.0 cannot be raised to a negative power")
		lower = self._protect_nan(
			min(
				pow(a[0],b[0]),
				pow(a[0],b[1]),
				pow(a[1],b[0]),
				pow(a[1],b[1])),
			'lower')
		
		upper = self._protect_nan(
			max(
				pow(a[0],b[0]),
				pow(a[0],b[1]),
				pow(a[1],b[0]),
				pow(a[1],b[1])),
			'upper')

		return (lower,upper)

	def _min(self,a,b):
		"""
		Get the minimum of two confidence intervals

		Parameters
		----------
		a : tuple
			Confidence interval like: (lower,upper)
		b : tuple
			Confidence interval like: (lower,upper)
		"""        
		lower = min(a[0],b[0])
		upper = min(a[1],b[1])
		return (lower,upper)

	def _max(self,a,b):
		"""
		Get the maximum of two confidence intervals

		Parameters
		----------
		a : tuple
			Confidence interval like: (lower,upper)
		b : tuple
			Confidence interval like: (lower,upper)
		"""        
		lower = max(a[0],b[0])
		upper = max(a[1],b[1])
		return (lower,upper)

	def _abs(self,a):
		"""
		Absolute value of a confidence interval

		Parameters
		----------
		a : tuple
			Confidence interval like: (lower,upper)
		"""
		abs_a0 = abs(a[0])
		abs_a1 = abs(a[1])
		
		lower = self._protect_nan(
			min(abs_a0,abs_a1) \
			if np.sign(a[0])==np.sign(a[1]) else 0,
			'lower')

		upper = self._protect_nan(
			max(abs_a0,abs_a1),
			'upper')

		return (lower,upper)

	def _exp(self,a):
		"""
		Exponentiate a confidence interval

		Parameters
		----------
		a : tuple
			Confidence interval like: (lower,upper)
		"""
		
		
		lower = self._protect_nan(
			np.exp(a[0]),
			'lower')

		upper = self._protect_nan(
			np.exp(a[1]),
			'upper')

		return (lower,upper)

	def reset_base_node_dict(self,reset_data=False):
		""" 
		Reset base node dict to initial state 
		

		Parameters
		----------
		reset_data : bool
			Whether to reset the cached data 
			for each base node. This is needed less frequently
			than one needs to reset the bounds.
		"""
		for node_name in self.base_node_dict:
			self.base_node_dict[node_name]['bound_computed'] = False
			self.base_node_dict[node_name]['value_computed'] = False
			self.base_node_dict[node_name]['value'] = None
			self.base_node_dict[node_name]['lower'] = float('-inf')
			self.base_node_dict[node_name]['upper'] = float('inf')
			if reset_data:
				self.base_node_dict[node_name]['data_dict'] = None
				self.base_node_dict[node_name]['datasize'] = 0

		return
		
	def make_viz(self,title):
		""" 
		Make a graphviz diagram from a root node

		Parameters
		----------
		title : str
			The title you want to display at the top
			of the graph
		"""
		graph=graphviz.Digraph()
		graph.attr(label=title+'\n\n')
		graph.attr(labelloc='t')
		graph.node(str(self.root.index),label=self.root.__repr__(),
			shape='box',
			fontsize=f'{self.node_fontsize}')
		self.make_viz_helper(self.root,graph)
		return graph

	def make_viz_helper(self,root,graph):
		""" 
		Helper function for make_viz()
		Recurses through the parse tree
		and adds nodes and edges to the graph

		Parameters
		----------
		root : nodes.Node class instance
			root of the parse tree
		graph: graphviz.Digraph class instance
			The graphviz graph object
		"""
		if root.left:
			if root.left.node_type == 'base_node':
				style = 'filled'
				fillcolor='green'
			elif root.left.node_type == 'constant_node':
				style = 'filled'
				fillcolor='yellow'
			else:
				style = ''
				fillcolor='white'

			graph.node(str(root.left.index),str(root.left.__repr__()),
				style=style,fillcolor=fillcolor,shape='box',
				fontsize=f'{self.node_fontsize}')
			graph.edge(str(root.index),str(root.left.index))
			self.make_viz_helper(root.left,graph)

		if root.right:
			if root.right.node_type == 'base_node':
				style = 'filled'
				fillcolor='green'
			elif root.right.node_type == 'constant_node':
				style = 'filled'
				fillcolor='yellow'
			else:
				style = ''
				fillcolor='white'
			graph.node(str(root.right.index),str(root.right.__repr__()),
				style=style,fillcolor=fillcolor,shape='box',
				fontsize=f'{self.node_fontsize}')
			graph.edge(str(root.index),str(root.right.index))
			self.make_viz_helper(root.right,graph)   
