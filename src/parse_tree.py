import ast
import graphviz
import numpy as np
from operator import itemgetter
from .stats_utils import *
from functools import reduce

# Special functions that are always leaf nodes
special_functions = [
	'Mean_Error','Mean_Squared_Error','Pr', 'FPR','TPR','FNR','TNR'
]

# map ast operators to string representations of operators
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
	ast.FloorDiv: '//'
}

class Node(object):
	""" 
	The base class for all parse tree nodes
	
	Attributes
	----------
	name : str
		the name of the node
	index : int
		the index of the node in the tree, root index is 0
	left : Node object or None
		left child node
	right : Node object or None
		right child node
	lower : float
		lower confidence bound
	upper : float
		upper confidence bound
	**kwargs : 
		optional additional arguments which get bundled into a dict

	Methods
	-------
	__repr__()
		String representation of the object 

	"""
	def __init__(self,name,lower,upper,**kwargs):
		self.name = name
		self.index = None 
		self.left  = None 
		self.right = None 
		self.lower = lower 
		self.upper = upper 

	def __repr__(self):
		lower_bracket = '['
		upper_bracket = ']'
		if np.isinf(self.lower):
			lower_bracket = '('
		if np.isinf(self.upper):
			upper_bracket = ')'

		bounds_str = f'{lower_bracket}{self.lower:g}, {self.upper:g}{upper_bracket}' \
			if (self.lower or self.upper) else '()'

		return '\n'.join(
			[
				'['+str(self.index)+']',
				str(self.name),
				u'\u03B5' + ' ' + bounds_str
			]
		) 
  
class BaseNode(Node):
	""" 
	Class for base variable leaf nodes
	in the parse tree.
	Inherits all attributes from Node class

	
	Attributes
	----------
	name : str
		The name of the node
	node_type : str
		equal to 'base_node'
	lower : float
		Lower confidence bound
	upper : float
		Upper confidence bound
	delta : float
		The share of the confidence put into this node
	compute_lower : bool
		Whether to compute the lower confidence interval
	compute_upper : bool
		Whether to compute the upper confidence interval
	conditional_columns: List(str)
		When calculating confidence bounds on a special 
		function, condition on these columns being == 1

	Methods
	-------
	calculate_bounds(bound_method)
		Calculate confidence bounds given a method, such as t-test
	
	compute_HC_lowerbound()
		--TODO--
		Calculate high confidence lower bound. 

	compute_HC_upperbound()
		--TODO--
		Calculate high confidence upper bound
	
	compute_HC_upper_and_lowerbound()
		--TODO--
		Calculate high confidence upper and lower bound

	"""
	def __init__(self,
		name,
		lower=float('-inf'),
		upper=float('inf'),
		conditional_columns=[],
		**kwargs):
		"""
		Parameters
		----------
		name : str
			The name of the node
		lower : float
			The lower bound, default -infinity
		upper : float
			The upper bound, default infinity
		"""

		super().__init__(name,lower,upper,**kwargs)
		self.node_type = 'base_node'
		self.delta = 0 
		self.is_special = False 
		self.special_function_name = '' # non-empty if is_special True
		self.compute_lower = True 
		self.compute_upper = True 
		self.conditional_columns = conditional_columns

	def __repr__(self):
		""" 
		Overrides Node.__repr__()
		"""
		return super().__repr__() + ', ' + u'\u03B4' + f'={self.delta:g}'
	
	def mask_dataframe(self,dataset,conditional_columns):
		"""
		"""
		masks = reduce(np.logical_and,(dataset.df[col]==1 for col in conditional_columns))
		masked_df = dataset.df.loc[masks] 
		return masked_df

	def calculate_data_forbound(self,**kwargs):
		theta,dataset,model = itemgetter(
					'theta','dataset','model')(kwargs)

		if kwargs['branch'] == 'candidate_selection':
			# Then we're in candidate selection
			n_safety = kwargs['n_safety']

		# mask the data using the conditional columns, if present
		if self.conditional_columns:
			dataframe = self.mask_dataframe(
				dataset,self.conditional_columns)
		else:
			dataframe = dataset.df

		# If in candidate selection want to use safety data size
		# in bound calculation
		if kwargs['branch'] == 'candidate_selection':
			frac_masked = len(dataframe)/len(dataset.df)
			datasize = int(round(frac_masked*n_safety))
		else:
			datasize = len(dataframe)
		# Separate features from label
		label_column = dataset.label_column
		labels = dataframe[label_column]
		features = dataframe.loc[:, dataframe.columns != label_column]
		features.insert(0,'offset',1.0) # inserts a column of 1's

		# drop sensitive column names
		if dataset.sensitive_column_names:
			features = features.drop(columns=dataset.sensitive_column_names)
		data_dict = {'features':features,'labels':labels}  
		return data_dict,datasize

	def calculate_bounds(self,
		**kwargs):
		"""
		Parameters
		----------
		method : str
			The method for calculating the bounds, 
			default Student's t-test
		""" 
		if 'bound_method' in kwargs:
			bound_method = kwargs['bound_method']
			if bound_method == 'manual':
				# Bounds set by user
				return self.lower,self.upper

			elif bound_method == 'random':
				# Randomly assign lower and upper bounds
				lower, upper = (
					np.random.randint(0,2),
					np.random.randint(2,4)
					)
				return lower,upper

			else:
				# Real confidence bound 

				# --TODO-- abstract away to support things like 
				# getting confidence intervals from bootstrap
				# and RL cases
				branch = kwargs['branch']
				model = kwargs['model']
				theta = kwargs['theta']
				data_dict = kwargs['data_dict']
				# take samples from the special function   
				special_samples = model.sample_from_statistic(
					self.special_function_name,theta,data_dict)

				if self.compute_lower and self.compute_upper:
					if branch == 'candidate_selection':
						lower,upper = self.predict_HC_upper_and_lowerbound(
							data=special_samples,
							delta=self.delta,
							**kwargs)  
					elif branch == 'safety_test':
						lower,upper = self.compute_HC_upper_and_lowerbound(
							data=special_samples,
							delta=self.delta,
							**kwargs)  
				else:
					raise NotImplementedError(
						"Have not implemented one-sided confidence bounds yet")

		return lower,upper

	def compute_HC_lowerbound(self,
		data,
		datasize,
		delta,
		**kwargs):
		"""
		Parameters
		----------
		-- TODO -- 
		""" 
		if 'bound_method' in kwargs:
			bound_method = kwargs['bound_method']
			if bound_method == 'ttest':	
				lower = data.mean() - stddev(data) / np.sqrt(datasize) * tinv(1.0 - delta, datasize - 1)
			else:
				raise NotImplementedError(f"Bounding method {bound_method} is not supported yet")
			
		return lower

	def compute_HC_upperbound(self,
		data,
		datasize,
		delta,
		**kwargs):
		"""
		Parameters
		----------
		-- TODO -- 
		"""
		if 'bound_method' in kwargs:
			bound_method = kwargs['bound_method']
			if bound_method == 'ttest':
				upper = data.mean() + stddev(data) / np.sqrt(datasize) \
					* tinv(1.0 - delta, datasize - 1)
			else:
				raise NotImplementedError("Have not implemented" 
					f"confidence bounds with bound_method: {bound_method}")
			
		return upper
	
	def compute_HC_upper_and_lowerbound(self,
		data,
		datasize,
		delta,
		**kwargs):
		"""
		Parameters
		----------
		-- TODO -- 
		"""
		if 'bound_method' in kwargs:
			bound_method = kwargs['bound_method']
			if bound_method == 'ttest':
				lower = self.compute_HC_lowerbound(data=data,
					datasize=datasize,delta=delta/2,
					**kwargs)
				upper = self.compute_HC_upperbound(data=data,
					datasize=datasize,delta=delta/2,
					**kwargs)

			elif bound_method == 'manual':
				pass
			else:
				raise NotImplementedError("Have not implemented" 
					f"confidence bounds with bound_method: {bound_method}")
		else:
			raise NotImplementedError("Have not implemented" 
					"confidence bounds without the keyword bound_method")

		return lower,upper
  
	def predict_HC_lowerbound(self,
		data,
		datasize,
		delta,
		**kwargs):
		"""
		Parameters
		----------
		-- TODO -- 
		""" 
		# print(data.shape)
		# print(delta)
		if 'bound_method' in kwargs:
			bound_method = kwargs['bound_method']

			if bound_method == 'ttest':
				lower = data.mean() - 2*stddev(data) / np.sqrt(datasize) * tinv(1.0 - delta, datasize - 1)
			else:
				raise NotImplementedError(f"Bounding method {bound_method} is not supported yet")
			
		return lower

	def predict_HC_upperbound(self,
		data,
		datasize,
		delta,
		**kwargs):
		"""
		Parameters
		----------
		-- TODO -- 
		""" 
		# print(data.shape)
		# print(delta)
		if 'bound_method' in kwargs:
			bound_method = kwargs['bound_method']
			if bound_method == 'ttest':
				lower = data.mean() + 2*stddev(data) / np.sqrt(datasize) * tinv(1.0 - delta, datasize - 1)
			else:
				raise NotImplementedError(f"Bounding method {bound_method} is not supported yet")
			
		return lower

	def predict_HC_upper_and_lowerbound(self,
		data,
		datasize,
		delta,
		conditional_columns=[],
		**kwargs):
		"""
		Parameters
		----------
		-- TODO -- 
		"""
		if 'bound_method' in kwargs:
			bound_method = kwargs['bound_method']
			if bound_method == 'ttest':
				lower = self.predict_HC_lowerbound(data=data,
					datasize=datasize,delta=delta/2,
					**kwargs)
				upper = self.predict_HC_upperbound(data=data,
					datasize=datasize,delta=delta/2,
					**kwargs)

			elif bound_method == 'manual':
				pass
			else:
				raise NotImplementedError(f"Have not implemented" 
					"confidence bounds with bound_method: {bound_method}")
			
		return lower,upper


class ConstantNode(Node):
	""" 
	Class for constant leaf nodes 
	in the parse tree. 
	Inherits all attributes from Node class

	Attributes
	----------
	name : str
		The name of the node
	value: float
		The value of the constant the node represents
	node_type : str
		'constant_node'

	"""
	def __init__(self,name,value,**kwargs):
		"""
		Sets lower and upper bound as the value of 
		the constant

		Parameters
		----------
		name : str
			The name of the node
		value: float
			The value of the constant 
		"""
		super().__init__(name=name,
			lower=value,upper=value,**kwargs)
		self.value = value
		self.node_type = 'constant_node'
  

class InternalNode(Node):
	""" 
	Class for internal (non-leaf) nodes 
	in the parse tree.
	These represent operators, such as +,-,*,/ etc.
	Inherits all attributes from Node class

	Attributes
	----------
	name : str
		The name of the node
	"""
	def __init__(self,name,
		lower=float('-inf'),upper=float('inf'),**kwargs):
		super().__init__(name,lower,upper,**kwargs)
		self.node_type = 'internal_node'


class ParseTree(object):
	""" 
	Class to represent a parse tree for a single behavioral constraint

	Attributes
	----------
	name : root
		Root node which contains the whole tree 
		via left and right attributes.
		Gets assigned when tree is built
	delta: float
		Confidence level. Specifies the maximum probability 
		that the algorithm can return a solution violates the
		behavioral constraint.
	n_nodes: int
		Total number of nodes in the parse tree
	n_base_nodes: int
		Number of base variable nodes in the parse tree.
		Does not include constants.
	base_node_dict: dict
		Keeps track of base variable nodes,
		their confidence bounds and whether 
		the bounds have been calculated
		for a given base node already.
		Helpful for handling case where we have 
		duplicate base nodes 
	node_fontsize: int
		Fontsize used in nodes displayed with graphviz 

	Methods
	-------
	create_from_ast(s)
		Create the node structure of the tree
		given a mathematical string expression, s

	_ast_tree_helper(root)
		Helper function for create_from_ast()

	_ast2pt_node(ast_node)
		Mapper between python's ast library's
		node objects to our node objects

	assign_deltas(weight_method)
		Assign the delta values to the base nodes in the tree

	_assign_deltas_helper(node,weight_method)
		Helper function for assign_deltas()

	propagate_bounds(bound_method='ttest')
		Traverse the parse tree, calculate confidence
		bounds on base nodes and 
		then propagate bounds using propagation logic

	_propagator_helper(node,bound_method)
		Helper function for propagate_bounds()

	_protect_nan(bound,bound_type)
		Handle nan as negative infinity if in lower bound
		and postitive infinity if in upper bound 

	_propagate(node)
		Given an internal node, calculate 
		the propagated confidence interval
		from its children using the 
		node's operator type

	add(a,b)
		Add intervals a and b

	sub(a,b)
		Subtract intervals a and b

	mult(a,b)
		Multiply intervals a and b

	div(a,b)
		Divide intervals a and b    

	abs(a)
		Take the absolute value of interval a 

	exp(a)
		Calculate e raised to the interval a 

	make_viz(title)
		Make a graphviz graph object of 
		the parse tree and give it a title

	make_viz_helper(root,graph)
		Helper function for make_viz()

	"""
	def __init__(self,delta):
		self.root = None 
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
		self.node_index = 0

		tree = ast.parse(s)
		# makes sure this is a single expression
		assert len(tree.body) == 1 

		expr = tree.body[0]
		root = expr.value

		# Recursively build the tree
		self.root = self._ast_tree_helper(root)

	def _ast_tree_helper(self,node):
		""" 
		From a given node in the ast tree,
		make a node in our tree and recurse
		to children of this node.

		Attributes
		----------
		node : ast.AST node class instance 
			
		"""
		# base case
		if node is None:
			return None

		# make a new node object
		new_node,is_leaf = self._ast2pt_node(node)

		if new_node.node_type == 'base_node':
			self.n_base_nodes += 1

			# check if special function
			# strip out conditional columns and parentheses
			node_name_isolated = new_node.name.split(
				"|")[0].strip().strip('(').strip()
			
			if node_name_isolated in special_functions:
				new_node.is_special = True
				new_node.special_function_name = node_name_isolated

			# if node with this name not already in self.base_node_dict
			# then make a new entry 
			if new_node.name not in self.base_node_dict:
				# 
				self.base_node_dict[new_node.name] = {
					'computed':False,
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
			return new_node

		if hasattr(node,'left'):
			new_node.left = self._ast_tree_helper(node.left)
		if hasattr(node,'right'):
			new_node.right = self._ast_tree_helper(node.right)
		if hasattr(node,'args') and node.func.id not in special_functions:
			for ii,arg in enumerate(node.args):
				new_node.left = self._ast_tree_helper(arg)

		return new_node

	def _ast2pt_node(self,ast_node):
		""" 
		Mapper to convert ast.AST node objects
		to our Node() objects

		Parameters
		----------
		ast_node : ast.AST node class instance
		"""
		is_leaf = False
		kwargs = {}
		conditional_columns = []
		if isinstance(ast_node,ast.BinOp):
			if ast_node.op.__class__ == ast.BitOr:
				# BitOr is used for "X | Y" i.e. "X given Y" 
				node_class = BaseNode

				conditional_columns = [x.id for x in ast_node.right.elts]
				# node_name = ast_node.left.id
				node_name = '(' + ' | '.join([ast_node.left.id,str(conditional_columns)]) + ')'
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

		elif isinstance(ast_node,ast.Name):
			# If variable name is "e" then make it a constant, not a base variable
			if ast_node.id == 'e':
				node_name = 'e'
				node_class = ConstantNode
				node_value = np.e
				is_leaf = True
				return node_class(node_name,node_value),is_leaf
			else:
				node_class = BaseNode
				node_name = ast_node.id
			is_leaf = True

		elif isinstance(ast_node,ast.Constant):
			node_class = ConstantNode
			node_value = ast_node.value
			node_name = str(node_value)
			is_leaf = True
			return node_class(node_name,node_value),is_leaf

		elif isinstance(ast_node,ast.Call):
			node_class = InternalNode
			node_name = ast_node.func.id

		return node_class(node_name),is_leaf

	def assign_deltas(self,weight_method='equal',**kwargs):
		""" 
		Assign the delta values to the base nodes in the tree.

		Parameters
		----------
		weight_method : str
			How you want to assign the deltas to the base nodes
			'equal' : split up delta equally among base nodes 
		"""
		assert self.n_nodes > 0, "Number of nodes must be > 0"
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
		weight_method : str
			How you want to assign the deltas to the base nodes
				'equal' : split up delta equally among base nodes 
		"""
		
		if not node:
			return

		if node.node_type == 'base_node':
			if weight_method == 'equal':
				node.delta = self.delta/self.n_base_nodes

		self._assign_deltas_helper(node.left,weight_method)
		self._assign_deltas_helper(node.right,weight_method)
		return

	def propagate_bounds(self,
		**kwargs):
		""" 
		Postorder traverse (left, right, root)
		through the tree and calculate confidence
		bounds on base nodes using a specified bound_method,
		then propagate bounds using propagation logic

		Parameters
		----------
		bound_method : str
			The method for calculating confidence bounds 
				'ttest' : Student's t test
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
		bound_method : str
			The method for calculating confidence bounds 
				'ttest' : Student's t test
		"""

		# if we hit a constant node or run past the end of the tree
		# return because we don't need to calculate bounds
		if not node or isinstance(node,ConstantNode):
			return 

		# if we hit a BaseNode,
		# then calculate confidence bounds and return 
		if isinstance(node,BaseNode):
			# Check if bound has already been calculated this node name
			# If so, use precalculated bound
			if self.base_node_dict[node.name]['computed'] == True:
				# print("Bound already computed for this node name")
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

				node.lower,node.upper = node.calculate_bounds(
					**kwargs)
				self.base_node_dict[node.name]['computed'] = True
				self.base_node_dict[node.name]['lower'] = node.lower
				self.base_node_dict[node.name]['upper'] = node.upper
			return 
		
		# traverse to children first
		self._propagator_helper(node.left,
			**kwargs)
		self._propagator_helper(node.right,
			**kwargs)
		
		# Here we must be at an internal node and therefore need to propagate
		node.lower,node.upper = self._propagate(node)
	
	def _protect_nan(self,bound,bound_type):
		""" 
		Handle nan as negative infinity if in lower bound
		and postitive infinity if in upper bound 

		Parameters
		----------
		bound : float
			Upper or lower bound 
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
		node : Node() class instance
		"""
		if node.name == 'add':
			a = (node.left.lower,node.left.upper)
			b = (node.right.lower,node.right.upper)
			return self.add(a,b)
			
		if node.name == 'sub':
			a = (node.left.lower,node.left.upper)
			b = (node.right.lower,node.right.upper)
			return self.sub(a,b)
			
		if node.name == 'mult':
			a = (node.left.lower,node.left.upper)
			b = (node.right.lower,node.right.upper)
			return self.mult(a,b)

		if node.name == 'div':
			a = (node.left.lower,node.left.upper)
			b = (node.right.lower,node.right.upper)
			return self.div(a,b) 
		
		if node.name == 'pow':
			a = (node.left.lower,node.left.upper)
			b = (node.right.lower,node.right.upper)
			return self.pow(a,b)

		if node.name == 'abs':
			# takes one node
			a = (node.left.lower,node.left.upper)
			return self.abs(a)
		elif node.name == 'exp':
			# takes one node
			a = (node.left.lower,node.left.upper)
			return self.exp(a)

		else:
			raise NotImplementedError("Encountered an operation we do not yet support", node.name)
	
	def add(self,a,b):
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

	def sub(self,a,b):
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

	def mult(self,a,b):
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

	def div(self,a,b):
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
			lower,upper = self.mult(a,new_b)

		elif b[0] == 0:
			# reduces to multiplication of a*(1/b[1],+inf)
			new_b = (1/b[1],float('inf'))
			lower,upper = self.mult(a,new_b)
		else:
			# b is either entirely negative or positive
			# reduces to multiplication of a*(1/b[1],1/b[0])
			new_b = (1/b[1],1/b[0])
			lower, upper = self.mult(a,new_b)

		return (lower,upper)

	def abs(self,a):
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

	def exp(self,a):
		"""
		Exponentiate a confidence interval
		--TODO-- make this pow(A,B) where 
		A and B can both be intervals or scalars

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

	def pow(self,a,b):
		"""
		Get the confidence interval on 
		pow(A,B) where 
		A and B are both be intervals 

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

	def reset_base_node_dict(self,reset_data=False):
		""" 
		Reset base node dict to initial state 
		This is all that should
		be necessary before each successive 
		propagation.

		"""
		for node_name in self.base_node_dict:
			self.base_node_dict[node_name]['computed'] = False
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
		graph.node(str(self.root.index),self.root.__repr__(),
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
		root : Node() class instance
			root of the parse tree
		graph: graphviz.Digraph() class instance
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


