"""
Main module for building parse trees from behavioral constraints
"""

import ast
import warnings

import graphviz
import autograd.numpy as np  # Thinly-wrapped version of Numpy

from seldonian.warnings.custom_warnings import *
from .nodes import *
from .operators import *

default_bound_method = "ttest"


class ParseTree(object):
    def __init__(self, delta, regime, sub_regime, columns=[]):
        """
        Class to represent a parse tree for a single behavioral constraint

        :param delta:
                Confidence level. Specifies the maximum probability
                that the algorithm can return a solution violat the
                behavioral constraint.
        :type delta: float
        :param regime: The category of the machine learning algorithm,
                e.g., supervised_learning or reinforcement_learning
        :type regime: str
        :param sub_regime: The sub-category of ml algorithm, e.g.
                classification or regression for supervised learning.
                Use 'all' for RL.
        :type sub_regime: str
        :param columns: The names of the columns in the dataframe.
                Used to determine if conditional columns provided by user
                are appropriate.
        :type columns: List(str)
        :ivar root:
                Root node which contains the whole tree
                via left and right child attributes.
                Gets assigned when tree is built
        :vartype root: nodes.Node object
        :ivar constraint_str:
                The string expression for the behavioral
                constraint
        :vartype constraint_str: str
        :ivar n_nodes:
                Total number of nodes in the parse tree
        :vartype n_nodes: int
        :ivar n_base_nodes:
                Number of base variable nodes in the parse tree.
                Does not include constants. If a base variable,
                such as PR | [M] appears more than once in the
                constraint_str each appearance contributes
                to n_base_nodes
        :vartype n_base_nodes: int
        :ivar base_node_dict:
                Keeps track of unique base variable nodes,
                their confidence bounds and whether
                the bounds have been calculated
                for a given base node already.
                Helpful for handling case where we have
                duplicate base nodes
        :vartype base_node_dict: dict
        :ivar node_fontsize:
                Fontsize used for graphviz visualizations
        :vartype node_fontsize: int
        :ivar available_measure_functions:
                A list of measure functions for the
                given regime and sub-regime, e.g. "Mean_Error"
                for supervised regression or "PR", i.e. Positive Rate
                for supervised classification.
        :vartype available_measure_functions: int
        """
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0,1)")
        self.delta = delta
        self.regime = regime
        self.sub_regime = sub_regime
        self.columns = columns
        self.root = None
        self.constraint_str = ""
        self.n_nodes = 0
        self.n_base_nodes = 0
        self.base_node_dict = {}
        self.node_fontsize = 12
        self.available_measure_functions = measure_functions_dict[self.regime][
            self.sub_regime
        ]

    def build_tree(self, constraint_str, delta_weight_method="equal"):
        """
        Convenience function for building the tree from
        a constraint string,
        weighting of deltas to each base node, and
        assigning which nodes need upper and lower bounding

        :param constraint_str:
                mathematical expression written in Python syntax
                from which we build the parse tree
        :type constraint_str: str
        :param delta_weight_method: str,
                How you want to assign the deltas to the base nodes.
                The default 'equal' splits up delta equally
                among unique base nodes
        :type delta_weight_method: str, defaults to 'equal'
        """

        self.create_from_ast(s=constraint_str)

        self.assign_deltas(weight_method=delta_weight_method)

        self.assign_bounds_needed()

    def create_from_ast(self, s):
        """
        Create the node structure of the tree
        given a mathematical string expression, s

        :param s:
                mathematical expression written in Python syntax
                from which we build the parse tree
        :type s: str
        """
        # Preprocessing string
        preprocessed_s = self._preprocess_constraint_str(s)
        self.constraint_str = preprocessed_s
        self.node_index = 0

        tree = ast.parse(preprocessed_s)
        # makes sure this is a single expression
        assert len(tree.body) == 1

        expr = tree.body[0]
        root = expr.value

        # Recursively build the tree
        self.root = self._ast_tree_helper(root)

    def _preprocess_constraint_str(self, s):
        """
        Check if inequalities present and
        move everything to one side so final
        constraint string is in the form: {constraint_str} <= 0

        Also does some validation checks to make sure string
        that was passed is valid

        :param s:
                mathematical expression written in Python syntax
                from which we build the parse tree
        :type s: str
        :return: String for g
        :rtype: str
        """
        if "<=" in s:
            assert s.count("<=") == 1
            assert s.count(">=") == 0
            start_index = s.index("<=")
            LHS = s[0:start_index].strip()
            RHS = s[start_index + 2 :].strip()
            if RHS == "0":
                new_s = LHS
            else:
                new_s = LHS + f"-({RHS})"
        elif ">=" in s:
            assert s.count(">=") == 1
            assert s.count("<=") == 0
            start_index = s.index(">=")
            LHS = s[:start_index].strip()
            RHS = s[start_index + 2 :].strip()
            if LHS == "0":
                new_s = RHS
            else:
                new_s = RHS + f"-({LHS})"
        else:
            new_s = s

        # Validate that new string does not have bad symbols in it
        for c in ["<", ">", "="]:
            if c in new_s:
                raise NotImplementedError(
                    "Error parsing your expression."
                    " An operator was used which we do not support: "
                    f"{c}"
                )
        return new_s

    def _ast_tree_helper(self, ast_node):
        """
        From a given node in the ast tree,
        make a node in the tree and recurse
        to children of this node.

        :param ast_node: node in the ast tree
        :type ast_node: ast.AST node object
        """
        # base case
        if ast_node is None:
            return None

        is_parent = False

        # handle unary operator like "-var"
        if isinstance(ast_node, ast.UnaryOp):
            # Only handle unary "-", reject rest
            if ast_node.op.__class__ != ast.USub:
                op = not_supported_op_mapper[ast_node.op.__class__]
                raise NotImplementedError(
                    "Error parsing your expression."
                    " A unary operator was used which we do not support: "
                    f"{op}"
                )

            # If operand is a constant, make a ConstantNode
            # with a negative value
            if isinstance(ast_node.operand, ast.Constant):
                node_value = -ast_node.operand.value
                node_name = str(-ast_node.operand.value)
                is_leaf = True
                new_node = ConstantNode(node_name, node_value)
            else:
                # Make three nodes, -1, * and whatever the operand is
                new_node_parent = InternalNode("mult")
                self.n_nodes += 1
                new_node_parent.index = self.node_index
                self.node_index += 1

                new_node_parent.left = ConstantNode("-1", -1.0)
                self.n_nodes += 1
                new_node_parent.left.index = self.node_index
                self.node_index += 1

                new_node, is_leaf = self._ast2pt_node(ast_node.operand)
                new_node_parent.right = new_node
                new_node_parent.right.index = self.node_index
                is_parent = True
                ast_node = ast_node.operand

        else:
            new_node, is_leaf = self._ast2pt_node(ast_node)

        if isinstance(new_node, BaseNode):
            self.n_base_nodes += 1

            # strip out conditional columns and parentheses
            # to get the measure function name
            # does not fail if none are present
            node_name_isolated = (
                new_node.name.split("|")[0].split("_[")[0].strip().strip("(").strip()
            )

            if (
                node_name_isolated not in self.available_measure_functions
                and node_name_isolated not in custom_base_node_dict
            ):
                raise NotImplementedError(
                    "Error parsing your expression. "
                    "A variable name was used which we do not recognize: "
                    f"{node_name_isolated}"
                )
            new_node.measure_function_name = node_name_isolated

            # if node with this name not already in self.base_node_dict
            # then make a new entry
            if new_node.name not in self.base_node_dict:
                self.base_node_dict[new_node.name] = {
                    "bound_method": default_bound_method,
                    "bound_computed": False,
                    "value_computed": False,
                    "lower": float("-inf"),
                    "upper": float("inf"),
                    "data_dict": None,
                    "datasize": 0,
                }

        self.n_nodes += 1
        new_node.index = self.node_index
        self.node_index += 1

        # If node is a leaf node, don't check for children
        if is_leaf:
            if is_parent:
                return new_node_parent
            return new_node
        # otherwise we are at an internal node
        # and need to recurse
        if hasattr(ast_node, "left"):
            new_node.left = self._ast_tree_helper(ast_node.left)
        if hasattr(ast_node, "right"):
            new_node.right = self._ast_tree_helper(ast_node.right)

        # Handle functions like min(), abs(), etc...
        if hasattr(ast_node, "args") and (
            ast_node.func.id not in self.available_measure_functions
        ):
            if len(ast_node.args) == 0:
                raise RuntimeError(
                    "Please check the syntax of the function: "
                    f" {new_node.name}()."
                    " It appears you provided no arguments"
                )
            elif len(ast_node.args) > 2:
                raise RuntimeError(
                    "Please check the syntax of the function:"
                    f" {new_node.name}()."
                    " It appears you provided more than two arguments"
                )
            if ast_node.func.id in ["abs", "exp", "log"] and len(ast_node.args) > 1:
                raise RuntimeError(
                    "Please check the syntax of the function:"
                    f" {new_node.name}()."
                    " It appears you provided more than one argument"
                )
            if ast_node.func.id in ["min", "max"] and len(ast_node.args) == 1:
                raise RuntimeError(
                    "Please check the syntax of the function: "
                    f"{new_node.name}(). "
                    "This function must take two arguments."
                )
            for ii, arg in enumerate(ast_node.args):
                if ii == 0:
                    new_node.left = self._ast_tree_helper(arg)
                if ii == 1:
                    new_node.right = self._ast_tree_helper(arg)

        if is_parent:
            return new_node_parent
        return new_node

    def _ast2pt_node(self, ast_node):
        """
        From ast.AST node object, create
        one of the node objects from :py:mod:`.Nodes`

        :param ast_node: node in the ast tree
        :type ast_node: ast.AST node object
        """
        is_leaf = False
        kwargs = {}

        if isinstance(ast_node, ast.Tuple):
            raise RuntimeError(
                "Error parsing your expression."
                " The issue is most likely due to"
                " missing/mismatched parentheses or square brackets"
                " in a conditional expression involving '|'."
            )

        if isinstance(ast_node, ast.BinOp):
            # +,-,*,/,**,| operators
            if ast_node.op.__class__ == ast.BitOr:
                # BitOr is the "|" operator, used to represent
                # a "A | B" -> "A given B"

                node_class = BaseNode
                node_kwargs = {}

                try:
                    conditional_columns = [str(x.id) for x in ast_node.right.elts]
                    conditional_columns_liststr = (
                        "[" + ",".join(conditional_columns) + "]"
                    )
                    if isinstance(ast_node.left, ast.Subscript):
                        node_class, left_node_kwargs = self._parse_subscript(
                            ast_node.left
                        )
                        left_id = left_node_kwargs["name"]
                        if node_class.__name__ == "ConfusionMatrixBaseNode":
                            node_kwargs["cm_true_index"] = left_node_kwargs[
                                "cm_true_index"
                            ]
                            node_kwargs["cm_pred_index"] = left_node_kwargs[
                                "cm_pred_index"
                            ]
                        elif node_class.__name__ == "RLAltRewardBaseNode":
                            node_kwargs["alt_reward_number"] = left_node_kwargs[
                                "alt_reward_number"
                            ]
                        else:
                            node_kwargs["class_index"] = left_node_kwargs["class_index"]
                    else:
                        left_id = ast_node.left.id
                except:
                    raise RuntimeError(
                        "Error parsing your expression."
                        " The issue is most likely due to"
                        " missing/mismatched parentheses or square brackets"
                        " in a conditional expression involving '|'."
                    )

                # Make sure conditional columns provided are valid
                for col in conditional_columns:
                    if col not in self.columns:
                        raise RuntimeError(
                            "A column provided in your constraint str: "
                            f"{col} was not in the list of "
                            f" columns provided: {self.columns}"
                        )
                node_kwargs["conditional_columns"] = conditional_columns
                node_name = " | ".join([left_id, conditional_columns_liststr])

                node_kwargs["name"] = node_name

                is_leaf = True

                return node_class(**node_kwargs), is_leaf
            else:
                node_class = InternalNode
                try:
                    node_name = op_mapper[ast_node.op.__class__]
                except KeyError:
                    op = not_supported_op_mapper[ast_node.op.__class__]
                    raise NotImplementedError(
                        "Error parsing your expression."
                        " An operator was used which we do not support: "
                        f"{op}"
                    )
                return node_class(node_name), is_leaf

        elif isinstance(ast_node, ast.Subscript):
            node_class, node_kwargs = self._parse_subscript(ast_node)
            is_leaf = True
            return node_class(**node_kwargs), is_leaf

        elif isinstance(ast_node, ast.Name):
            # named quantity like "e", "Mean_Squared_Error"
            # Custom base nodes will be caught here too
            # If variable name is "e" then make it a constant, not a base variable
            if ast_node.id == "e":
                node_name = "e"
                node_class = ConstantNode
                node_value = np.e
                is_leaf = True
                return node_class(node_name, node_value), is_leaf
            else:
                if ast_node.id in custom_base_node_dict:
                    # A user-defined base node
                    node_class = custom_base_node_dict[ast_node.id]
                    node_name = ast_node.id

                elif ast_node.id not in self.available_measure_functions:
                    raise NotImplementedError(
                        "Error parsing your expression."
                        " A variable name was used which we do not recognize: "
                        f"{ast_node.id}"
                    )
                else:
                    # a measure function in our list
                    node_class = BaseNode
                    node_name = ast_node.id

                is_leaf = True
                return node_class(node_name), is_leaf

        elif isinstance(ast_node, ast.Constant):
            # A constant floating point or integer number
            node_class = ConstantNode
            node_value = ast_node.value
            node_name = str(node_value)
            is_leaf = True
            return node_class(node_name, node_value), is_leaf

        elif isinstance(ast_node, ast.Call):
            # a function call like abs(arg1), min(arg1,arg2)
            node_class = InternalNode
            node_name = ast_node.func.id

        return node_class(node_name), is_leaf

    def _parse_subscript(self, ast_node):
        if ast_node.value.id not in [
            "CM_",
            "PR_",
            "NR_",
            "FPR_",
            "TNR_",
            "TPR_",
            "FNR_",
            "J_pi_new_"
        ]:
            raise NotImplementedError(
                "Error parsing your expression."
                " A subscript was used in a way we do not support: "
                f"{ast_node.value.id}"
            )
        if ast_node.value.id == "CM_":
            # This is a confusion matrix element
            node_class = ConfusionMatrixBaseNode
            # ast API changed after Python 3.8 in how it handles slices
            try:
                # >= 3.9 syntax
                elements = ast_node.slice.elts
            except AttributeError:
                # 3.8 syntax
                elements = ast_node.slice.value.elts

            assert len(elements) == 2
            row_index, col_index = [x.value for x in elements]
            node_name = f"CM_[{row_index},{col_index}]"
            node_kwargs = {}
            node_kwargs["name"] = node_name
            node_kwargs["cm_true_index"] = row_index
            node_kwargs["cm_pred_index"] = col_index
        elif ast_node.value.id == "J_pi_new_":
            # alternate reward function
            node_class = RLAltRewardBaseNode
            try:
                # 3.8 syntax
                alt_reward_number = ast_node.slice.value.value
            except AttributeError:
                alt_reward_number = ast_node.slice.value
            node_name = f"{ast_node.value.id}[{alt_reward_number}]"
            node_kwargs = {}
            node_kwargs["name"] = node_name
            node_kwargs["alt_reward_number"] = alt_reward_number
        else:
            # It's one of the PR_[i], FPR_[i], etc. functions
            node_class = MultiClassBaseNode
            # ast API changed after Python 3.8 in how it handles slices
            try:
                # 3.8 syntax
                class_index = ast_node.slice.value.value
            except AttributeError:
                class_index = ast_node.slice.value
            assert type(class_index) == int
            node_name = f"{ast_node.value.id}[{class_index}]"
            node_kwargs = {}
            node_kwargs["name"] = node_name
            node_kwargs["class_index"] = class_index

        return node_class, node_kwargs

    def assign_deltas(self, weight_method="equal", **kwargs):
        """
        Assign the delta values to the base nodes in the tree.

        :param weight_method: str, defaults to 'equal'
                How you want to assign the deltas to the base nodes.
                The default 'equal' splits up delta equally
                among unique base nodes
        :type weight_method: str
        """
        assert self.n_base_nodes > 0, (
            "Number of base nodes must be > 0."
            " Make sure to build the tree before assigning deltas."
        )
        self._assign_deltas_helper(self.root, weight_method, **kwargs)

    def _assign_deltas_helper(self, node, weight_method, **kwargs):
        """
        Helper function to traverse the parse tree
        and assign delta values to base nodes.

        :param node: node in the parse tree
        :type node: :py:class:`.Node` object
        :param weight_method:
                How you want to assign the deltas to the base nodes
        :type weight_method: str
        """

        if not node:
            return

        if isinstance(node, BaseNode):  # captures all child classes of BaseNode as well
            if weight_method == "equal":
                node.delta = self.delta / len(self.base_node_dict)

        self._assign_deltas_helper(node.left, weight_method)
        self._assign_deltas_helper(node.right, weight_method)
        return

    def assign_bounds_needed(self, **kwargs):
        """
        Breadth first search through the tree and
        decide which bounds are required to compute
        on each child node. Eventually we get to base nodes.
        There are cases where it is not always
        necessary to compute both lower and upper
        bounds because at the end all we care about
        is the upper bound of the root node.
        """
        assert self.n_nodes > 0, "Number of nodes must be > 0"
        # initialize needed bounds for root
        lower_needed = False
        upper_needed = True
        self._assign_bounds_helper(self.root, lower_needed, upper_needed, **kwargs)

    def _assign_bounds_helper(self, node, lower_needed, upper_needed, **kwargs):
        """
        Helper function to traverse the parse tree
        and assign which bounds we need to calculate
        on the base nodes.

        :param node: node in the parse tree
        :type node: :py:class:`.Node` object
        :param lower_needed:
                Whether lower bound needs to be calculated
        :type lower_needed: bool
        :param upper_needed:
                Whether upper bound needs to be calculated
        :type upper_needed: bool
        """

        # if we go off the end return
        if not node:
            return
        node.will_lower_bound = lower_needed
        node.will_upper_bound = upper_needed

        # If we get to a base node or constant node, then return
        if isinstance(node, BaseNode) or isinstance(node, ConstantNode):
            return

        if isinstance(node, InternalNode):
            # depending on operator type and current bounds
            # needed in the parent, determine which bounds
            # need to be calculated on the child nodes

            bounds_dict = bounds_required_dict[node.name]

            two_children = True
            if len(bounds_dict["lower"]) == 2:
                two_children = False

            if lower_needed and upper_needed:
                if two_children:
                    (
                        left_lower_needed,
                        left_upper_needed,
                        right_lower_needed,
                        right_upper_needed,
                    ) = np.logical_or(bounds_dict["lower"], bounds_dict["upper"])
                else:
                    (left_lower_needed, left_upper_needed) = np.logical_or(
                        bounds_dict["lower"], bounds_dict["upper"]
                    )

            elif lower_needed or upper_needed:
                # only one bound is needed
                if lower_needed:
                    if two_children:
                        (
                            left_lower_needed,
                            left_upper_needed,
                            right_lower_needed,
                            right_upper_needed,
                        ) = bounds_dict["lower"]
                    else:
                        (left_lower_needed, left_upper_needed) = bounds_dict["lower"]

                if upper_needed:
                    if two_children:
                        (
                            left_lower_needed,
                            left_upper_needed,
                            right_lower_needed,
                            right_upper_needed,
                        ) = bounds_dict["upper"]
                    else:
                        (left_lower_needed, left_upper_needed) = bounds_dict["upper"]
            else:
                raise RuntimeError("Need at least lower or upper bound")

            self._assign_bounds_helper(node.left, left_lower_needed, left_upper_needed)

            if two_children:
                self._assign_bounds_helper(
                    node.right, right_lower_needed, right_upper_needed
                )
            return

    def propagate_bounds(self, **kwargs):
        """
        Postorder traverse (left, right, root)
        through the tree and calculate confidence
        bounds on base nodes,
        then propagate bounds using propagation logic
        """
        if not self.root:
            return []

        self._propagator_helper(self.root, **kwargs)

    def _propagator_helper(self, node, **kwargs):
        """
        Helper function for traversing
        through the tree and propagating confidence bounds

        :param node: node in the parse tree
        :type node: :py:class:`.Node` object
        """

        # if we hit a constant node or run past the end of the tree
        # return because we don't need to calculate bounds
        if not node or isinstance(node, ConstantNode):
            return

        # if we hit a BaseNode,
        # then calculate confidence bounds and return
        if isinstance(node, BaseNode):
            # Check if bound has already been calculated for this node name
            # If so, use precalculated bound
            if self.base_node_dict[node.name]["bound_computed"] == True:
                node.lower = self.base_node_dict[node.name]["lower"]
                node.upper = self.base_node_dict[node.name]["upper"]
                return
            else:
                # Need to calculate the bound
                if "dataset" in kwargs:
                    # Check if data has already been prepared
                    # for this node name. If so, use precalculated data
                    if self.base_node_dict[node.name]["data_dict"] != None:
                        data_dict = self.base_node_dict[node.name]["data_dict"]
                        datasize = self.base_node_dict[node.name]["datasize"]
                    else:
                        # Data not prepared already. Need to do that.
                        if isinstance(node, RLAltRewardBaseNode):
                            kwargs["alt_reward_number"] = node.alt_reward_number
                        data_dict, datasize = node.calculate_data_forbound(**kwargs)
                        self.base_node_dict[node.name]["data_dict"] = data_dict
                        self.base_node_dict[node.name]["datasize"] = datasize

                    kwargs["data_dict"] = data_dict
                    kwargs["datasize"] = datasize

                bound_method = self.base_node_dict[node.name]["bound_method"]
                if isinstance(node, ConfusionMatrixBaseNode):
                    kwargs["cm_true_index"] = node.cm_true_index
                    kwargs["cm_pred_index"] = node.cm_pred_index
                bound_result = node.calculate_bounds(
                    bound_method=bound_method, **kwargs
                )
                self.base_node_dict[node.name]["bound_computed"] = True

                if node.will_lower_bound:
                    node.lower = bound_result["lower"]
                    self.base_node_dict[node.name]["lower"] = node.lower

                if node.will_upper_bound:
                    node.upper = bound_result["upper"]
                    self.base_node_dict[node.name]["upper"] = node.upper

            return

        # traverse to children first
        self._propagator_helper(node.left, **kwargs)
        self._propagator_helper(node.right, **kwargs)

        # Here we must be at an internal node and therefore need to propagate
        node.lower, node.upper = self.propagate(node)

    def evaluate_constraint(self, **kwargs):
        """
        Evaluate the constraint itself (not bounds)
        Postorder traverse (left, right, root)
        through the tree and calculate the values
        of the base nodes
        then propagate bounds using propagation logic
        """
        if not self.root:
            return []

        self._evaluator_helper(self.root, **kwargs)

    def _evaluator_helper(self, node, **kwargs):
        """
        Helper function for traversing
        through the tree to evaluate the constraint

        :param node: node in the parse tree
        :type node: :py:class:`.Node` object
        """

        # if we hit a constant node or run past the end of the tree
        # return because we don't need to calculate anything
        if not node or isinstance(node, ConstantNode):
            return

        # if we hit a BaseNode,
        # then calculate the value and return
        if isinstance(node, BaseNode):
            # Check if value has already been calculated for this node name
            # If so, use precalculated value
            if self.base_node_dict[node.name]["value_computed"] == True:
                node.value = self.base_node_dict[node.name]["value"]
                return
            else:
                if "dataset" in kwargs:
                    # Check if data has already been prepared
                    # for this node name. If so, use precalculated data
                    if self.base_node_dict[node.name]["data_dict"] != None:
                        data_dict = self.base_node_dict[node.name]["data_dict"]
                        datasize = self.base_node_dict[node.name]["datasize"]
                    else:
                        print("Calculating data for bound")
                        if isinstance(node, RLAltRewardBaseNode):
                            kwargs["alt_reward_number"] = node.alt_reward_number
                        data_dict, datasize = node.calculate_data_forbound(**kwargs)
                        self.base_node_dict[node.name]["data_dict"] = data_dict
                        self.base_node_dict[node.name]["datasize"] = datasize

                    kwargs["data_dict"] = data_dict
                    kwargs["datasize"] = datasize

                if isinstance(node, ConfusionMatrixBaseNode):
                    kwargs["cm_true_index"] = node.cm_true_index
                    kwargs["cm_pred_index"] = node.cm_pred_index
                
                value = node.calculate_value(**kwargs)
                node.value = value
                self.base_node_dict[node.name]["value_computed"] = True
                self.base_node_dict[node.name]["value"] = node.value

            return

        # traverse to children first
        self._evaluator_helper(node.left, **kwargs)
        self._evaluator_helper(node.right, **kwargs)

        # Here we must be at an internal node and therefore need to propagate
        node.value = self._propagate_value(node)

    def _propagate_value(self, node):
        """
        Helper function for propagating values

        :param node: node in the parse tree
        :type node: :py:class:`.Node` object
        """
        a = node.left.value
        if node.right:
            b = node.right.value

        if node.name == "add":
            return a + b

        if node.name == "sub":
            return a - b

        if node.name == "mult":
            return a * b

        if node.name == "div":
            return a / b

        if node.name == "pow":
            warning_msg = (
                "Warning: Power operation "
                "is an experimental feature. Use with caution."
            )
            return pow(a, b)

        if node.name == "min":
            return min(a, b)

        if node.name == "max":
            return max(a, b)

        if node.name == "abs":
            # takes one node
            return abs(a)

        if node.name == "exp":
            # takes one node
            return np.exp(a)

        if node.name == "log":
            # takes one node
            return np.log(a)

        else:
            raise NotImplementedError(
                "Encountered an operation we do not yet support", node.name
            )

    def _protect_nan(self, bound, bound_type):
        """
        Handle nan as negative infinity if in lower bound
        and postitive infinity if in upper bound

        :param bound:
                The value of the upper or lower bound
        :type bound: float
        :param bound_type:
                'lower' or 'upper'
        :type bound_type: str
        """
        if np.isnan(bound):
            if bound_type == "lower":
                return float("-inf")
            if bound_type == "upper":
                return float("inf")
        else:
            return bound

    def propagate(self, node):
        """
        Helper function for propagating confidence bounds

        :param node: node in the parse tree
        :type node: :py:class:`.Node` object
        """
        if node.name == "add":
            a = (node.left.lower, node.left.upper)
            b = (node.right.lower, node.right.upper)
            return self._add(a, b)

        if node.name == "sub":
            a = (node.left.lower, node.left.upper)
            b = (node.right.lower, node.right.upper)
            return self._sub(a, b)

        if node.name == "mult":
            a = (node.left.lower, node.left.upper)
            b = (node.right.lower, node.right.upper)
            return self._mult(a, b)

        if node.name == "div":
            a = (node.left.lower, node.left.upper)
            b = (node.right.lower, node.right.upper)
            return self._div(a, b)

        if node.name == "pow":
            warning_msg = (
                "Warning: Power operation "
                "is an experimental feature. Use with caution."
            )
            warnings.warn(warning_msg)
            a = (node.left.lower, node.left.upper)
            b = (node.right.lower, node.right.upper)
            return self._pow(a, b)

        if node.name == "min":
            a = (node.left.lower, node.left.upper)
            b = (node.right.lower, node.right.upper)
            return self._min(a, b)

        if node.name == "max":
            a = (node.left.lower, node.left.upper)
            b = (node.right.lower, node.right.upper)
            return self._max(a, b)

        if node.name == "abs":
            # takes one node
            a = (node.left.lower, node.left.upper)
            return self._abs(a)

        if node.name == "exp":
            # takes one node
            a = (node.left.lower, node.left.upper)
            return self._exp(a)

        if node.name == "log":
            # takes one node
            a = (node.left.lower, node.left.upper)
            return self._log(a)

        else:
            raise NotImplementedError(
                "Encountered an operation we do not yet support", node.name
            )

    def _add(self, a, b):
        """
        Add two confidence intervals

        :param a:
                Confidence interval like: (lower,upper)
        :type a: tuple
        :param b:
                Confidence interval like: (lower,upper)
        :type b: tuple
        """
        lower = self._protect_nan(a[0] + b[0], "lower")

        upper = self._protect_nan(a[1] + b[1], "upper")

        return (lower, upper)

    def _sub(self, a, b):
        """
        Subract two confidence intervals

        :param a:
                Confidence interval like: (lower,upper)
        :type a: tuple
        :param b:
                Confidence interval like: (lower,upper)
        :type b: tuple
        """
        lower = self._protect_nan(a[0] - b[1], "lower")

        upper = self._protect_nan(a[1] - b[0], "upper")

        return (lower, upper)

    def _mult(self, a, b):
        """
        Multiply two confidence intervals

        :param a:
                Confidence interval like: (lower,upper)
        :type a: tuple
        :param b:
                Confidence interval like: (lower,upper)
        :type b: tuple
        """
        lower = self._protect_nan(
            min(a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]), "lower"
        )

        upper = self._protect_nan(
            max(a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]), "upper"
        )

        return (lower, upper)

    def _div(self, a, b):
        """
        Divide two confidence intervals

        :param a:
                Confidence interval like: (lower,upper)
        :type a: tuple
        :param b:
                Confidence interval like: (lower,upper)
        :type b: tuple
        """

        if b[0] < 0 < b[1]:
            # unbounded
            lower = float("-inf")
            upper = float("inf")

        elif b[1] == 0:
            # reduces to multiplication of a*(-inf,1/b[0]]
            new_b = (float("-inf"), 1 / b[0])
            lower, upper = self._mult(a, new_b)

        elif b[0] == 0:
            # reduces to multiplication of a*(1/b[1],+inf)
            new_b = (1 / b[1], float("inf"))
            lower, upper = self._mult(a, new_b)
        else:
            # b is either entirely negative or positive
            # reduces to multiplication of a*(1/b[1],1/b[0])
            new_b = (1 / b[1], 1 / b[0])
            lower, upper = self._mult(a, new_b)

        return (lower, upper)

    def _pow(self, a, b):
        """
        Get the confidence interval on
        pow(a,b) where
        b and b are both be intervals

        :param a:
                Confidence interval like: (lower,upper)
        :type a: tuple
        :param b:
                Confidence interval like: (lower,upper)
        :type b: tuple
        """

        # First, cases that are not allowed
        if a[0] < 0:
            raise ArithmeticError(
                f"Cannot compute interval: pow({a},{b}) because first argument contains negatives"
            )
        if 0 in a and (b[0] < 0 or b[1] < 1):
            raise ZeroDivisionError("0.0 cannot be raised to a negative power")
        lower = self._protect_nan(
            min(pow(a[0], b[0]), pow(a[0], b[1]), pow(a[1], b[0]), pow(a[1], b[1])),
            "lower",
        )

        upper = self._protect_nan(
            max(pow(a[0], b[0]), pow(a[0], b[1]), pow(a[1], b[0]), pow(a[1], b[1])),
            "upper",
        )

        return (lower, upper)

    def _min(self, a, b):
        """
        Get the minimum of two confidence intervals

        :param a:
                Confidence interval like: (lower,upper)
        :type a: tuple
        :param b:
                Confidence interval like: (lower,upper)
        :type b: tuple
        """
        lower = min(a[0], b[0])
        upper = min(a[1], b[1])
        return (lower, upper)

    def _max(self, a, b):
        """
        Get the maximum of two confidence intervals

        :param a:
                Confidence interval like: (lower,upper)
        :type a: tuple
        :param b:
                Confidence interval like: (lower,upper)
        :type b: tuple
        """
        lower = max(a[0], b[0])
        upper = max(a[1], b[1])
        return (lower, upper)

    def _abs(self, a):
        """
        Absolute value of a confidence interval

        :param a:
                Confidence interval like: (lower,upper)
        :type a: tuple
        """
        abs_a0 = abs(a[0])
        abs_a1 = abs(a[1])

        lower = self._protect_nan(
            min(abs_a0, abs_a1) if np.sign(a[0]) == np.sign(a[1]) else 0, "lower"
        )

        upper = self._protect_nan(max(abs_a0, abs_a1), "upper")

        return (lower, upper)

    def _exp(self, a):
        """
        Exponentiate a confidence interval

        :param a:
                Confidence interval like: (lower,upper)
        :type a: tuple
        """

        lower = self._protect_nan(np.exp(a[0]), "lower")

        upper = self._protect_nan(np.exp(a[1]), "upper")

        return (lower, upper)

    def _log(self,a):
        """
        Take log of a confidence interval

        :param a:
                Confidence interval like: (lower,upper)
        :type a: tuple
        """
        lower = self._protect_nan(np.log(a[0]), "lower")

        upper = self._protect_nan(np.log(a[1]), "upper")

        return (lower, upper)

    def reset_base_node_dict(self, reset_data=False):
        """
        Reset base node dict to initial obs

        :param reset_data:
                Whether to reset the cached data
                for each base node. This is needed less frequently
                than one needs to reset the bounds.
        :type reset_data: bool
        """
        for node_name in self.base_node_dict:
            self.base_node_dict[node_name]["bound_computed"] = False
            self.base_node_dict[node_name]["value_computed"] = False
            self.base_node_dict[node_name]["value"] = None
            self.base_node_dict[node_name]["lower"] = float("-inf")
            self.base_node_dict[node_name]["upper"] = float("inf")
            if reset_data:
                self.base_node_dict[node_name]["data_dict"] = None
                self.base_node_dict[node_name]["datasize"] = 0

        return

    def make_viz(self, title):
        """
        Make a graphviz diagram from a root node

        :param title:
                The title you want to display at the top
                of the graph
        :type title: str
        """
        graph = graphviz.Digraph()
        graph.attr(label=title + "\n\n")
        graph.attr(labelloc="t")
        graph.node(
            str(self.root.index),
            label=self.root.__repr__(),
            shape="box",
            fontsize=f"{self.node_fontsize}",
        )
        self.make_viz_helper(self.root, graph)
        return graph

    def make_viz_helper(self, root, graph):
        """
        Helper function for make_viz()
        Recurses through the parse tree
        and adds nodes and edges to the graph

        :param root:
                root of the parse tree
        :type root: :py:class:`.Node` object
        :param graph:
                The graphviz graph object
        :type graph: graphviz.Digraph object
        """
        if root.left:
            if root.left.node_type == "base_node":
                style = "filled"
                fillcolor = "green"
            elif root.left.node_type == "constant_node":
                style = "filled"
                fillcolor = "yellow"
            else:
                style = ""
                fillcolor = "white"

            graph.node(
                str(root.left.index),
                str(root.left.__repr__()),
                style=style,
                fillcolor=fillcolor,
                shape="box",
                fontsize=f"{self.node_fontsize}",
            )
            graph.edge(str(root.index), str(root.left.index))
            self.make_viz_helper(root.left, graph)

        if root.right:
            if root.right.node_type == "base_node":
                style = "filled"
                fillcolor = "green"
            elif root.right.node_type == "constant_node":
                style = "filled"
                fillcolor = "yellow"
            else:
                style = ""
                fillcolor = "white"
            graph.node(
                str(root.right.index),
                str(root.right.__repr__()),
                style=style,
                fillcolor=fillcolor,
                shape="box",
                fontsize=f"{self.node_fontsize}",
            )
            graph.edge(str(root.index), str(root.right.index))
            self.make_viz_helper(root.right, graph)


def make_parse_trees_from_constraints(
    constraint_strs,
    deltas,
    regime="supervised_learning",
    sub_regime="regression",
    columns=[],
    delta_weight_method="equal",
):
    """
    Convenience function for creating parse trees
    given constraint strings and deltas

    :param constraint_strs: List of constraint strings
    :param deltas: List of deltas corresponding to each constraint
    :param regime: The category of the machine learning algorithm,
            e.g., supervised_learning or reinforcement_learning
    :type regime: str, defaults to "supervised_learning"
    :param sub_regime: The sub-category of the machine learning algorithm,
            e.g., classifiction or regression
    :type sub_regime: str, defults to "regression"
    :param columns: list of columns in the dataset, needed
            if constraints condition on any of these columns
    :param delta_weight_method: The method for weighting deltas
            across the base nodes.
    :type delta_weight_method: str, defults to "equal"
    """
    parse_trees = []
    for ii in range(len(constraint_strs)):
        constraint_str = constraint_strs[ii]

        delta = deltas[ii]

        # Create parse tree object
        pt = ParseTree(
            delta=delta, regime=regime, sub_regime=sub_regime, columns=columns
        )

        # Fill out tree
        pt.build_tree(
            constraint_str=constraint_str, delta_weight_method=delta_weight_method
        )

        parse_trees.append(pt)

    return parse_trees
