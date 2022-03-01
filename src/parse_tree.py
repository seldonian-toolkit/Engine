import ast
import graphviz
import numpy as np

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

class Node(object):
    """ 
    The base class for all parse tree nodes
    
    Attributes
    ----------
    name : str
        the name of the node
    index : int
        the index of the node in the tree, root index is 0
    left : int
        left child node
    right : int
        right child node
    lower : float
        lower confidence bound
    upper : float
        upper confidence bound
    **kwargs : dict
        optional additional key value pairs

    Methods
    -------
    __repr__()
        String representation of the object 

    """
    def __init__(self,name,lower,upper,**kwargs):
        
        self.name = name
        self.index = None 
        self.left  = None 
        self.right = None # right child node
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
        'base_node'
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
    **kwargs : dict
        Optional additional key value pairs

    Methods
    -------
    calculate_bounds(bound_method)
        Calculate confidence bounds given a method
    
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
        self.compute_lower = True # whether to compute lower bound
        self.compute_upper = True # whether to computer upper bound
        
        if 'conditional_columns' in kwargs:
            self.conditional_columns = kwargs['conditional_columns']
        else:
            self.conditional_columns = []

    def __repr__(self):
        """ 
        Overrides Node.__repr__()
        By adding the delta assigned to this node
        """
        node_repr = super().__repr__()
        return node_repr + ', ' + u'\u03B4' + f'={self.delta:g}'
     
    def calculate_bounds(self,bound_method='ttest',**kwargs):
        """
        Parameters
        ----------
        method : str
            The method for calculating the bounds, 
            default Student's t-test
        """ 

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

        elif self.compute_lower and self.compute_upper:
                lower,upper = self.compute_HC_upper_and_lowerbound(
                    self.delta,bound_method=bound_method,**kwargs)  
        else:
            raise NotImplementedError("Have not implemented one sided confidence bounds yet")

        return lower,upper

    def compute_HC_lowerbound(self,
        theta,X,Y,delta,
        conditional_columns=[],
        bound_method='ttest',
        conservative=False,
        **kwargs):
        """
        Parameters
        ----------
        -- TODO -- 
        """ 
        if bound_method == 'ttest':
            n  = len(data)
            if conservative:
                lower = data.mean() - 2*(stddev(data) / math.sqrt(n) * tinv(1.0 - delta, n - 1))
            else:
                lower = data.mean() - stddev(data) / math.sqrt(n) * tinv(1.0 - delta, n - 1)
        else:
            raise NotImplementedError(f"Bounding method {bound_method} is not supported yet")
            
        return lower

    def compute_HC_upperbound(self,
        theta,X,Y,delta,
        conditional_columns=[],
        bound_method='ttest',
        conservative=False,
        **kwargs):
        """
        Parameters
        ----------
        -- TODO -- 
        """
        if bound_method == 'ttest':
            n  = len(v)
            if conservative:
                upper = v.mean() + 2*(stddev(v) / math.sqrt(n) \
                    * tinv(1.0 - delta, n - 1))
            else:
                upper = v.mean() + stddev(v) / math.sqrt(n) \
                    * tinv(1.0 - delta, n - 1)
        else:
            raise NotImplementedError 
            
        return upper
    
    def compute_HC_upper_and_lowerbound(self,
        theta,X,Y,delta,
        conditional_columns=[],
        bound_method='ttest',
        conservative=False,
        **kwargs):
        """
        Parameters
        ----------
        -- TODO -- 
        """
        if bound_method == 'ttest':
            lower = self.compute_HC_lowerbound(delta/2, bound_method=bound_method,
                conservative=conservative **kwargs)
            upper = self.compute_HC_upperbound(delta/2, bound_method=bound_method,
                conservative=conservative **kwargs)
        elif bound_method == 'manual':
            pass
        else:
            raise NotImplementedError
            
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

            # check if has an entry in self.base_node_dict
            if new_node.name not in self.base_node_dict:
                self.base_node_dict[new_node.name] = {
                    'computed':False,
                    'lower':float('-inf'),
                    'upper':float('inf')
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
            # The | is the BitOr which denotes a conditional base node
            if ast_node.op.__class__ == ast.BitOr:
                node_class = BaseNode
                node_name = '(' + ' | '.join([ast_node.left.id,ast_node.right.id]) + ')'
                conditional_columns.append(ast_node.right.id)
                is_leaf = True
            else:
                node_class = InternalNode
                node_name = op_mapper[ast_node.op.__class__]

        elif isinstance(ast_node,ast.Name):
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

        if conditional_columns != []:
            kwargs['conditional_columns'] = conditional_columns

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

    def propagate_bounds(self,bound_method='ttest',**kwargs):
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
            bound_method=bound_method,**kwargs)
    
    def _propagator_helper(self,node,bound_method,**kwargs):
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
        # because we are at a leaf node
        if isinstance(node,BaseNode):
            # calculate bounds

            # Check if bound has already been calculated
            # If so, use precalculated bound
            if self.base_node_dict[node.name]['computed'] == True:
                # print(node)
                # print("Using precomputed bounds!")
                node.lower = self.base_node_dict[node.name]['lower']
                node.upper = self.base_node_dict[node.name]['upper'] 
                return
            else:
                node.lower,node.upper = node.calculate_bounds(
                    bound_method=bound_method)
                self.base_node_dict[node.name]['computed'] = True
                self.base_node_dict[node.name]['lower'] = node.lower
                self.base_node_dict[node.name]['upper'] = node.upper
            return 
        
        # traverse to children first
        self._propagator_helper(node.left,bound_method=bound_method)
        self._propagator_helper(node.right,bound_method=bound_method)
        
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


