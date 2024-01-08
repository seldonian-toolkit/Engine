import pytest
import time

from sklearn.model_selection import train_test_split

from seldonian.parse_tree.parse_tree import *
from seldonian.dataset import DataSetLoader, SupervisedDataSet
from seldonian.safety_test.safety_test import SafetyTest
from seldonian.utils.io_utils import load_json, load_pickle
from seldonian.models.models import LinearRegressionModel
from seldonian.dataset import RLDataSet, RLMetaData
from seldonian.RL.RL_model import RL_model


two_interval_options = [
    [[2.0, 3.0], [4.0, 5.0]],
    [[-2.0, 1.0], [2.0, 3.0]],
    [[0.5, 0.75], [2, 4]],
    [[0.5, 1.1], [0.0, 0.4]],
    [[-3.2, -2.0], [-4.5, -4.0]],
    [[-3.2, -2.0], [-4.5, 5.0]],
    [[-6.1, -6.0], [0.0, 0.5]],
    [[0.0, 0.5], [1.0, 9.0]],
    [[0.0, 0.5], [-6.8, -5.0]],
    [[float("-inf"), 5.0], [-6.8, -5.0]],
    [[0.0, float("inf")], [5.0, 10.0]],
    [[float("-inf"), float("inf")], [5.0, 10.0]],
    [[float("-inf"), float("inf")], [float("-inf"), float("inf")]],
    [[float("inf"), float("inf")], [float("inf"), float("inf")]],
]

single_interval_options = [
    [1, 2],
    [-3.2, -2.0],
    [-3.2, 2.0],
    [-5.1, 0],
    [0.0, 0.5],
    [float("-inf"), 0.0],
    [float("-inf"), 15342],
    [0.0, float("inf")],
    [float("-inf"), float("inf")],
    [float("inf"), float("inf")],
]

answer_dict = {
    "add": [
        [6.0, 8.0],
        [0.0, 4.0],
        [2.5, 4.75],
        [0.5, 1.5],
        [-7.7, -6.0],
        [-7.7, 3.0],
        [-6.1, -5.5],
        [1.0, 9.5],
        [-6.8, -4.5],
        [float("-inf"), 0.0],
        [5.0, float("inf")],
        [float("-inf"), float("inf")],
        [float("-inf"), float("inf")],
        [float("inf"), float("inf")],
    ],
    "sub": [
        [-3.0, -1.0],
        [-5.0, -1.0],
        [-3.5, -1.25],
        [0.1, 1.1],
        [0.8, 2.5],
        [-8.2, 2.5],
        [-6.6, -6.0],
        [-9.0, -0.5],
        [5.0, 7.3],
        [float("-inf"), 11.8],
        [-10.0, float("inf")],
        [float("-inf"), float("inf")],
        [float("-inf"), float("inf")],
        [float("-inf"), float("inf")],
    ],
    "mult": [
        [8.0, 15.0],
        [-6.0, 3.0],
        [1.0, 3.0],
        [0.0, 1.1 * 0.4],
        [8.0, 14.4],
        [-16.0, 14.4],
        [-3.05, 0.0],
        [0.0, 4.5],
        [-3.4, 0.0],
        [-34.0, float("inf")],
        [0.0, float("inf")],
        [float("-inf"), float("inf")],
        [float("-inf"), float("inf")],
        [float("inf"), float("inf")],
    ],
    "div": [
        [2 / 5.0, 3 / 4.0],
        [-1.0, 0.5],
        [1 / 8, 3 / 8],
        [0.5 / 0.4, float("inf")],
        [2 / 4.5, 3.2 / 4],
        [float("-inf"), float("inf")],
        [float("-inf"), -12.0],
        [0.0, 0.5],
        [-0.1, 0.0],
        [-1.0, float("inf")],
        [0.0, float("inf")],
        [float("-inf"), float("inf")],
        [float("-inf"), float("inf")],
        [float("-inf"), float("inf")],
    ],
    "pow": [
        [16, 243],
        [None, None],
        [pow(0.5, 4), pow(0.75, 2)],
        [pow(0.5, 0.4), pow(1.1, 0.4)],
        [None, None],  # input raises exception
        [None, None],  # input raises exception
        [None, None],  # input raises exception
        [0.0, 0.5],
        [0.0, 0.5],
        [None, None],  # input raises exception
        [0.0, float("inf")],
        [None, None],  # input raises exception
        [None, None],  # input raises exception
        [float("inf"), float("inf")],
    ],
    "min": [
        [2.0, 3.0],
        [-2.0, 1.0],
        [0.5, 0.75],
        [0.0, 0.4],
        [-4.5, -4.0],
        [-4.5, -2.0],
        [-6.1, -6.0],
        [0.0, 0.5],
        [-6.8, -5.0],
        [float("-inf"), -5.0],
        [0.0, 10.0],
        [float("-inf"), 10.0],
        [float("-inf"), float("inf")],
        [float("inf"), float("inf")],
    ],
    "max": [
        [4.0, 5.0],
        [2.0, 3.0],
        [2.0, 4.0],
        [0.5, 1.1],
        [-3.2, -2.0],
        [-3.2, 5.0],
        [0.0, 0.5],
        [1.0, 9.0],
        [0.0, 0.5],
        [-6.8, 5.0],
        [5.0, float("inf")],
        [5.0, float("inf")],
        [float("-inf"), float("inf")],
        [float("inf"), float("inf")],
    ],
    "abs": [
        [1.0, 2.0],
        [2.0, 3.2],
        [0, 3.2],
        [0, 5.1],
        [0, 0.5],
        [0, float("inf")],
        [0, float("inf")],
        [0, float("inf")],
        [0, float("inf")],
        [float("inf"), float("inf")],
    ],
    "log": [
        [0.0, np.log(2)],
        [float("-inf"), float("inf")],
        [float("-inf"), np.log(2)],
        [float("-inf"), float("-inf")],
        [float("-inf"), np.log(0.5)],
        [float("-inf"), float("-inf")],
        [float("-inf"), np.log(15342)],
        [float("-inf"), float("inf")],
        [float("-inf"), float("inf")],
        [float("inf"), float("inf")],
    ],
}

### Begin tests

########################
### Propagator tests ###
########################

@pytest.mark.parametrize("interval_index", range(len(two_interval_options)))
def test_add_bounds(interval_index, stump):
    ### Addition ###

    a, b = two_interval_options[interval_index]
    answer = answer_dict["add"][interval_index]
    pt = stump("add", a, b)
    pt.propagate_bounds()
    assert pt.root.lower == answer[0]
    assert pt.root.upper == answer[1]
    assert pt.base_node_dict["a"]["bound_computed"] == True
    assert pt.base_node_dict["b"]["bound_computed"] == True

@pytest.mark.parametrize("interval_index", range(len(two_interval_options)))
def test_subtract_bounds(interval_index, stump):
    ### Subtraction ###

    a, b = two_interval_options[interval_index]
    answer = answer_dict["sub"][interval_index]
    pt = stump("sub", a, b)
    pt.propagate_bounds()
    # Use approx due to floating point imprecision
    assert pt.root.lower == pytest.approx(answer[0])
    assert pt.root.upper == pytest.approx(answer[1])
    assert pt.base_node_dict["a"]["bound_computed"] == True
    assert pt.base_node_dict["b"]["bound_computed"] == True

@pytest.mark.parametrize("interval_index", range(len(two_interval_options)))
def test_multiply_bounds(interval_index, stump):
    ### Multiplication ###

    a, b = two_interval_options[interval_index]
    answer = answer_dict["mult"][interval_index]
    pt = stump("mult", a, b)
    pt.propagate_bounds()
    # Use approx due to floating point imprecision
    assert pt.root.lower == pytest.approx(answer[0])
    assert pt.root.upper == pytest.approx(answer[1])
    assert pt.base_node_dict["a"]["bound_computed"] == True
    assert pt.base_node_dict["b"]["bound_computed"] == True

@pytest.mark.parametrize("interval_index", range(len(two_interval_options)))
def test_divide_bounds(interval_index, stump):
    ### Division ###

    a, b = two_interval_options[interval_index]
    answer = answer_dict["div"][interval_index]
    pt = stump("div", a, b)
    pt.propagate_bounds()
    # Use approx due to floating point imprecision
    assert pt.root.lower == pytest.approx(answer[0])
    assert pt.root.upper == pytest.approx(answer[1])
    assert pt.base_node_dict["a"]["bound_computed"] == True
    assert pt.base_node_dict["b"]["bound_computed"] == True

@pytest.mark.parametrize("interval_index", range(len(two_interval_options)))
def test_power_bounds(interval_index, stump):
    ### power ###

    # A warning message should be raised
    # anytime the power operator is called
    warning_msg = (
        "Warning: Power operation " "is an experimental feature. Use with caution."
    )
    a, b = two_interval_options[interval_index]

    pt = stump("pow", a, b)
    if a[0] < 0:
        with pytest.warns(UserWarning, match=warning_msg):
            with pytest.raises(ArithmeticError) as excinfo:
                pt.propagate_bounds()

        assert "Cannot compute interval" in str(excinfo.value)
        assert "because first argument contains negatives" in str(excinfo.value)

    elif 0 in a and (b[0] < 0 or b[1] < 1):
        with pytest.warns(UserWarning, match=warning_msg):
            with pytest.raises(ZeroDivisionError) as excinfo:
                pt.propagate_bounds()

        assert "0.0 cannot be raised to a negative power" in str(excinfo.value)
    else:
        answer = answer_dict["pow"][interval_index]

        with pytest.warns(UserWarning, match=warning_msg):
            pt.propagate_bounds()

        # Use approx due to floating point imprecision
        assert pt.root.lower == pytest.approx(answer[0])
        assert pt.root.upper == pytest.approx(answer[1])
        assert pt.base_node_dict["a"]["bound_computed"] == True
        assert pt.base_node_dict["b"]["bound_computed"] == True

@pytest.mark.parametrize("interval_index", range(len(two_interval_options)))
def test_min_bounds(interval_index, stump):
    ### min ###

    a, b = two_interval_options[interval_index]

    pt = stump("min", a, b)

    answer = answer_dict["min"][interval_index]

    pt.propagate_bounds()
    # Use approx due to floating point imprecision
    assert pt.root.lower == pytest.approx(answer[0])
    assert pt.root.upper == pytest.approx(answer[1])
    assert pt.base_node_dict["a"]["bound_computed"] == True
    assert pt.base_node_dict["b"]["bound_computed"] == True

@pytest.mark.parametrize("interval_index", range(len(two_interval_options)))
def test_max_bounds(interval_index, stump):
    ### min ###

    a, b = two_interval_options[interval_index]

    pt = stump("max", a, b)

    answer = answer_dict["max"][interval_index]

    pt.propagate_bounds()
    # Use approx due to floating point imprecision
    assert pt.root.lower == pytest.approx(answer[0])
    assert pt.root.upper == pytest.approx(answer[1])
    assert pt.base_node_dict["a"]["bound_computed"] == True
    assert pt.base_node_dict["b"]["bound_computed"] == True

@pytest.mark.parametrize("interval_index", range(len(single_interval_options)))
def test_abs_bounds(interval_index, edge):
    ### Absolute value ###

    a = single_interval_options[interval_index]
    answer = answer_dict["abs"][interval_index]
    pt = edge("abs", a)
    pt.propagate_bounds()
    # Use approx due to floating point imprecision
    assert pt.root.lower == pytest.approx(answer[0])
    assert pt.root.upper == pytest.approx(answer[1])
    assert pt.base_node_dict["a"]["bound_computed"] == True

@pytest.mark.parametrize("interval_index", range(len(single_interval_options)))
def test_log_bounds(interval_index, edge):
    ### Absolute value ###

    a = single_interval_options[interval_index]
    answer = answer_dict["log"][interval_index]
    pt = edge("log", a)
    pt.propagate_bounds()
    # Use approx due to floating point imprecision
    assert pt.root.lower == pytest.approx(answer[0])
    assert pt.root.upper == pytest.approx(answer[1])
    assert pt.base_node_dict["a"]["bound_computed"] == True

##################
### Node tests ###
##################

def test_node_reprs(stump):
    a, b = [[2.0, 3.0], [4.0, 5.0]]

    pt = stump("add", a, b)
    pt.assign_bounds_needed()
    pt.assign_deltas()
    pt.propagate_bounds()

    # Before assigning which bounds are needed
    root_bounds_str = f"[_, 8]"
    assert pt.root.__repr__() == "\n".join(
        ["[0]", "add", "\u03B5" + " " + root_bounds_str]
    )
    left_bounds_str = f"[_, 3]"
    assert pt.root.left.__repr__() == "\n".join(
        ["[1]", "a", "\u03B5" + " " + left_bounds_str + ", \u03B4=(None,0.025)"]
    )

    right_bounds_str = f"[_, 5]"
    assert pt.root.right.__repr__() == "\n".join(
        ["[2]", "b", "\u03B5" + " " + right_bounds_str + ", \u03B4=(None,0.025)"]
    )
    # After assigning which bounds are needed
    pt = stump("add", a, b)

    pt.assign_bounds_needed()
    pt.assign_deltas()
    pt.propagate_bounds()

    # Before assigning which bounds are needed
    root_bounds_str = f"[_, 8]"
    assert pt.root.__repr__() == "\n".join(
        ["[0]", "add", "\u03B5" + " " + root_bounds_str]
    )
    left_bounds_str = f"[_, 3]"
    assert pt.root.left.__repr__() == "\n".join(
        ["[1]", "a", "\u03B5" + " " + left_bounds_str + ", \u03B4=(None,0.025)"]
    )

    right_bounds_str = f"[_, 5]"
    assert pt.root.right.__repr__() == "\n".join(
        ["[2]", "b", "\u03B5" + " " + right_bounds_str + ", \u03B4=(None,0.025)"]
    )

########################
### Parse tree tests ###
########################

def test_parse_tree_from_simple_string():
    constraint_str = "FPR - (FNR + PR)*4"
    delta = 0.05
    pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")
    pt.create_from_ast(constraint_str)
    assert pt.n_nodes == 7
    assert pt.n_base_nodes == 3
    assert len(pt.base_node_dict) == 3
    assert isinstance(pt.root, InternalNode)
    assert pt.root.name == "sub"
    assert pt.root.left.name == "FPR"
    assert pt.root.right.name == "mult"
    assert pt.root.right.left.name == "add"
    assert pt.root.right.left.left.name == "FNR"
    assert pt.root.right.left.right.name == "PR"
    assert pt.root.right.right.name == "4"
    assert pt.root.right.right.value == 4

def test_parse_tree_with_inequalities():
    # First one without inequalities
    # constraint_str = 'FPR <= 0.5 + 0.3*(PR | [M])'
    constraint_str = "FPR - (0.5 + (PR | [M]))"
    pt = ParseTree(
        delta=0.05,
        regime="supervised_learning",
        sub_regime="classification",
        columns=["M"],
    )

    # Fill out tree
    pt.build_tree(constraint_str=constraint_str, delta_weight_method="equal")

    assert pt.n_nodes == 5
    assert pt.n_base_nodes == 2
    assert len(pt.base_node_dict) == 2
    assert isinstance(pt.root, InternalNode)
    assert pt.root.name == "sub"
    assert pt.root.left.name == "FPR"
    assert pt.root.right.name == "add"
    assert pt.root.right.left.name == "0.5"
    assert pt.root.right.left.value == 0.5
    assert pt.root.right.right.name == "PR | [M]"

    # Now with <=
    constraint_str_lte = "FPR <= 0.5 + (PR | [M])"
    pt_lte = ParseTree(
        delta=0.05,
        regime="supervised_learning",
        sub_regime="classification",
        columns=["M"],
    )

    # Fill out tree
    pt_lte.build_tree(constraint_str=constraint_str_lte, delta_weight_method="equal")

    assert pt_lte.n_nodes == 5
    assert pt_lte.n_base_nodes == 2
    assert len(pt_lte.base_node_dict) == 2
    assert isinstance(pt_lte.root, InternalNode)
    assert pt_lte.root.name == "sub"
    assert pt_lte.root.left.name == "FPR"
    assert pt_lte.root.right.name == "add"
    assert pt_lte.root.right.left.name == "0.5"
    assert pt_lte.root.right.left.value == 0.5
    assert pt_lte.root.right.right.name == "PR | [M]"

    # Now with >=
    constraint_str_gte = "0.5 + (PR | [M]) >= FPR"
    pt_gte = ParseTree(
        delta=0.05,
        regime="supervised_learning",
        sub_regime="classification",
        columns=["M"],
    )

    # Fill out tree
    pt_gte.build_tree(constraint_str=constraint_str_gte, delta_weight_method="equal")

    assert pt_gte.n_nodes == 5
    assert pt_gte.n_base_nodes == 2
    assert len(pt_gte.base_node_dict) == 2
    assert isinstance(pt_gte.root, InternalNode)
    assert pt_gte.root.name == "sub"
    assert pt_gte.root.left.name == "FPR"
    assert pt_gte.root.right.name == "add"
    assert pt_gte.root.right.left.name == "0.5"
    assert pt_gte.root.right.left.value == 0.5
    assert pt_gte.root.right.right.name == "PR | [M]"

    # <= 0
    constraint_str_lte0 = "FPR - (0.5 + (PR | [M])) <= 0"
    pt_lte0 = ParseTree(
        delta=0.05,
        regime="supervised_learning",
        sub_regime="classification",
        columns=["M"],
    )

    # Fill out tree
    pt_lte0.build_tree(constraint_str=constraint_str_lte0, delta_weight_method="equal")

    assert pt_lte0.n_nodes == 5
    assert pt_lte0.n_base_nodes == 2
    assert len(pt_lte0.base_node_dict) == 2
    assert isinstance(pt_lte0.root, InternalNode)
    assert pt_lte0.root.name == "sub"
    assert pt_lte0.root.left.name == "FPR"
    assert pt_lte0.root.right.name == "add"
    assert pt_lte0.root.right.left.name == "0.5"
    assert pt_lte0.root.right.left.value == 0.5
    assert pt_lte0.root.right.right.name == "PR | [M]"

    # >= 0
    constraint_str_gte0 = "0 >= FPR - (0.5 + (PR | [M]))"
    pt_gte0 = ParseTree(
        delta=0.05,
        regime="supervised_learning",
        sub_regime="classification",
        columns=["M"],
    )

    # Fill out tree
    pt_gte0.build_tree(constraint_str=constraint_str_gte0, delta_weight_method="equal")

    assert pt_gte0.n_nodes == 5
    assert pt_gte0.n_base_nodes == 2
    assert len(pt_gte0.base_node_dict) == 2
    assert isinstance(pt_gte0.root, InternalNode)
    assert pt_gte0.root.name == "sub"
    assert pt_gte0.root.left.name == "FPR"
    assert pt_gte0.root.right.name == "add"
    assert pt_gte0.root.right.left.name == "0.5"
    assert pt_gte0.root.right.left.value == 0.5
    assert pt_gte0.root.right.right.name == "PR | [M]"

def test_math_functions():
    """Test that math functions like
    min(), max(), abs() and exp() get parsed
    as expected. min and max expect no more than two arguments."""
    constraint_str = "min((PR | [X]), (PR | [Y]))"
    delta = 0.05
    pt = ParseTree(
        delta,
        regime="supervised_learning",
        sub_regime="classification",
        columns=["X", "Y"],
    )
    pt.create_from_ast(constraint_str)

    assert pt.n_nodes == 3
    assert pt.n_base_nodes == 2
    assert len(pt.base_node_dict) == 2
    assert isinstance(pt.root, InternalNode)
    assert pt.root.name == "min"
    assert pt.root.left.name == "PR | [X]"
    assert pt.root.right.name == "PR | [Y]"

    constraint_str = "min((PR | [X]), (PR | [Y]), (PR | [Z]))"
    delta = 0.05
    pt = ParseTree(
        delta,
        regime="supervised_learning",
        sub_regime="classification",
        columns=["X", "Y", "Z"],
    )
    with pytest.raises(RuntimeError) as excinfo:
        pt.create_from_ast(constraint_str)

    error_str = (
        "Please check the syntax of the function: min()."
        " It appears you provided more than two arguments"
    )
    assert str(excinfo.value) == error_str

    constraint_str = "abs((PR | [X]), (PR | [Y]))"
    delta = 0.05
    pt = ParseTree(
        delta,
        regime="supervised_learning",
        sub_regime="classification",
        columns=["X", "Y", "Z"],
    )
    with pytest.raises(RuntimeError) as excinfo:
        pt.create_from_ast(constraint_str)

    error_str = (
        "Please check the syntax of the function: "
        f"abs(). "
        "It appears you provided more than one argument"
    )
    assert str(excinfo.value) == error_str

    constraint_str = "exp((PR | [X]), (PR | [Y]))"
    delta = 0.05
    pt = ParseTree(
        delta,
        regime="supervised_learning",
        sub_regime="classification",
        columns=["X", "Y", "Z"],
    )
    with pytest.raises(RuntimeError) as excinfo:
        pt.create_from_ast(constraint_str)

    error_str = (
        "Please check the syntax of the function: "
        f"exp(). "
        "It appears you provided more than one argument"
    )
    assert str(excinfo.value) == error_str

    constraint_str = "log((PR | [X]), (PR | [Y]))"
    delta = 0.05
    pt = ParseTree(
        delta,
        regime="supervised_learning",
        sub_regime="classification",
        columns=["X", "Y", "Z"],
    )
    with pytest.raises(RuntimeError) as excinfo:
        pt.create_from_ast(constraint_str)

    error_str = (
        "Please check the syntax of the function: "
        f"log(). "
        "It appears you provided more than one argument"
    )
    assert str(excinfo.value) == error_str

    constraint_str = "max((PR | [X]))"
    delta = 0.05
    pt = ParseTree(
        delta,
        regime="supervised_learning",
        sub_regime="classification",
        columns=["X", "Y", "Z"],
    )
    with pytest.raises(RuntimeError) as excinfo:
        pt.create_from_ast(constraint_str)

    error_str = (
        "Please check the syntax of the function: "
        f"max(). "
        "This function must take two arguments."
    )
    assert str(excinfo.value) == error_str

    constraint_str = "min((PR | [X]))"
    delta = 0.05
    pt = ParseTree(
        delta,
        regime="supervised_learning",
        sub_regime="classification",
        columns=["X", "Y", "Z"],
    )
    with pytest.raises(RuntimeError) as excinfo:
        pt.create_from_ast(constraint_str)

    error_str = (
        "Please check the syntax of the function: "
        f"min(). "
        "This function must take two arguments."
    )
    assert str(excinfo.value) == error_str

    constraint_str = "abs()"
    delta = 0.05
    pt = ParseTree(
        delta,
        regime="supervised_learning",
        sub_regime="classification",
        columns=["X", "Y", "Z"],
    )
    with pytest.raises(RuntimeError) as excinfo:
        pt.create_from_ast(constraint_str)

    error_str = (
        "Please check the syntax of the function:  abs()."
        " It appears you provided no arguments"
    )
    assert str(excinfo.value) == error_str

def test_measure_functions_recognized():
    delta = 0.05

    constraint_str = "Mean_Squared_Error - 2.0"
    pt = ParseTree(delta, regime="supervised_learning", sub_regime="regression")
    pt.create_from_ast(constraint_str)
    assert pt.root.left.measure_function_name == "Mean_Squared_Error"

    constraint_str = "(Mean_Error|[M]) - 2.0"

    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="regression", columns=["M"]
    )
    pt.create_from_ast(constraint_str)
    assert pt.root.left.measure_function_name == "Mean_Error"

    constraint_str = "(FPR|[A,B]) - 2.0"

    pt = ParseTree(
        delta,
        regime="supervised_learning",
        sub_regime="classification",
        columns=["A", "B"],
    )
    pt.create_from_ast(constraint_str)
    assert pt.root.left.measure_function_name == "FPR"

    # Test that a non-measure base node
    # is not recognized as measure
    constraint_str = "X - 2.0"

    pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")
    with pytest.raises(NotImplementedError) as excinfo:
        pt.create_from_ast(constraint_str)

    error_str = (
        "Error parsing your expression."
        " A variable name was used which we do not recognize: X"
    )
    assert str(excinfo.value) == error_str

    # Test that a non-measure base node
    # is not recognized as measure
    constraint_str = "(X | [A]) - 2.0"

    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="classification", columns=["A"]
    )
    with pytest.raises(NotImplementedError) as excinfo:
        pt.create_from_ast(constraint_str)

    error_str = (
        "Error parsing your expression."
        " A variable name was used which we do not recognize: X"
    )
    assert str(excinfo.value) == error_str

    constraint_str = "ACC >= 0.5"

    pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")
    pt.create_from_ast(constraint_str)
    assert pt.root.right.measure_function_name == "ACC"

    constraint_str = "(ACC | [A]) >= 0.5"

    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="classification", columns=["A"]
    )
    pt.create_from_ast(constraint_str)
    assert pt.root.right.measure_function_name == "ACC"

def test_multiclass_measure_functions():
    delta = 0.05
    constraint_str = "CM_[0,1] - 0.5"

    # Confusion matrix

    # Make sure error is raised if we use wrong sub_regime
    pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")
    with pytest.raises(NotImplementedError) as excinfo:
        pt.create_from_ast(constraint_str)
    error_str = (
        "Error parsing your expression. "
        "A variable name was used which we do not recognize: CM"
    )
    assert str(excinfo.value) == error_str

    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="multiclass_classification"
    )
    pt.create_from_ast(constraint_str)
    assert pt.root.left.measure_function_name == "CM"
    assert pt.root.left.name == "CM_[0,1]"
    assert pt.root.left.cm_true_index == 0
    assert pt.root.left.cm_pred_index == 1

    constraint_str = "(CM_[2,3] | [A,B]) - 0.5"
    pt = ParseTree(
        delta,
        regime="supervised_learning",
        sub_regime="multiclass_classification",
        columns=["A", "B"],
    )
    pt.create_from_ast(constraint_str)
    assert pt.root.left.measure_function_name == "CM"
    assert pt.root.left.name == "CM_[2,3] | [A,B]"
    assert pt.root.left.cm_true_index == 2
    assert pt.root.left.cm_pred_index == 3

    # PR, NR, FPR, TNR, TPR, FNR
    delta = 0.05
    for msr_func in ["PR", "NR", "FPR", "TNR", "TPR", "FNR"]:
        constraint_str = f"{msr_func}_[0]-0.5"

        pt = ParseTree(
            delta, regime="supervised_learning", sub_regime="multiclass_classification"
        )
        pt.create_from_ast(constraint_str)
        assert pt.root.left.measure_function_name == msr_func
        assert pt.root.left.name == f"{msr_func}_[0]"
        assert pt.root.left.class_index == 0

        constraint_str = f"({msr_func}_[1] | [A,B]) - 0.5"
        pt = ParseTree(
            delta,
            regime="supervised_learning",
            sub_regime="multiclass_classification",
            columns=["A", "B"],
        )
        pt.create_from_ast(constraint_str)
        assert pt.root.left.measure_function_name == msr_func
        assert pt.root.left.name == f"{msr_func}_[1] | [A,B]"
        assert pt.root.left.class_index == 1

    # Accuracy
    constraint_str = "ACC >= 0.5"

    # Make sure error is raised if we use wrong sub_regime
    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="multiclass_classification"
    )
    pt.create_from_ast(constraint_str)
    assert pt.root.right.measure_function_name == "ACC"

    constraint_str = "(ACC | [A]) >= 0.5"

    # Make sure error is raised if we use wrong sub_regime
    pt = ParseTree(
        delta,
        regime="supervised_learning",
        sub_regime="multiclass_classification",
        columns=["A"],
    )
    pt.create_from_ast(constraint_str)
    assert pt.root.right.measure_function_name == "ACC"

def test_rl_new_policy_base_node():
    delta = 0.05
    constraint_str = "J_pi_new_IS - 0.5"

    pt = ParseTree(delta, regime="reinforcement_learning", sub_regime="all")
    pt.create_from_ast(constraint_str)
    assert pt.root.left.measure_function_name == "J_pi_new_IS"
    assert pt.root.left.name == "J_pi_new_IS"
    assert isinstance(pt.root.left, NewPolicyPerformanceBaseNode)

def test_rl_alt_reward_string():
    delta = 0.05
    constraint_str = "J_pi_new_IS_[1] - 0.5"

    pt = ParseTree(delta, regime="reinforcement_learning", sub_regime="all")
    pt.create_from_ast(constraint_str)
    assert pt.root.left.measure_function_name == "J_pi_new_IS"
    assert pt.root.left.name == "J_pi_new_IS_[1]"
    assert pt.root.left.alt_reward_number == 1
    assert isinstance(pt.root.left, NewPolicyPerformanceBaseNode)
    assert isinstance(pt.root.left, RLAltRewardBaseNode)

    constraint_str = "(J_pi_new_IS_[2] | [A,B]) - 0.5"
    pt = ParseTree(
        delta, regime="reinforcement_learning", sub_regime="all", columns=["A", "B"]
    )
    pt.create_from_ast(constraint_str)
    assert pt.root.left.measure_function_name == "J_pi_new_IS"
    assert pt.root.left.name == "J_pi_new_IS_[2] | [A,B]"
    assert pt.root.left.alt_reward_number == 2
    assert isinstance(pt.root.left, NewPolicyPerformanceBaseNode)
    assert isinstance(pt.root.left, RLAltRewardBaseNode)

    constraint_str = "J_pi_new_PDIS_[1] - 0.5"

    pt = ParseTree(delta, regime="reinforcement_learning", sub_regime="all")
    pt.create_from_ast(constraint_str)
    assert pt.root.left.measure_function_name == "J_pi_new_PDIS"
    assert pt.root.left.name == "J_pi_new_PDIS_[1]"
    assert pt.root.left.alt_reward_number == 1
    assert isinstance(pt.root.left, NewPolicyPerformanceBaseNode)
    assert isinstance(pt.root.left, RLAltRewardBaseNode)

    constraint_str = "J_pi_new_WIS_[1] - 0.5"

    pt = ParseTree(delta, regime="reinforcement_learning", sub_regime="all")
    pt.create_from_ast(constraint_str)
    assert pt.root.left.measure_function_name == "J_pi_new_WIS"
    assert pt.root.left.name == "J_pi_new_WIS_[1]"
    assert pt.root.left.alt_reward_number == 1
    assert isinstance(pt.root.left, NewPolicyPerformanceBaseNode)
    assert isinstance(pt.root.left, RLAltRewardBaseNode)

def test_rl_alt_reward_bad_string():
    # Test that using non-numeric characters for the alt reward number raises an error
    delta = 0.05
    constraint_str = "J_pi_new_IS_[N] - 0.5"

    pt = ParseTree(delta, regime="reinforcement_learning", sub_regime="all")
    with pytest.raises(RuntimeError) as excinfo:
        pt.create_from_ast(constraint_str)
    error_str = "The alternate reward number you entered was not an integer."
    assert str(excinfo.value) == error_str

    constraint_str = "J_pi_new_IS_[1.1] - 0.5"

    pt = ParseTree(delta, regime="reinforcement_learning", sub_regime="all")
    with pytest.raises(RuntimeError) as excinfo:
        pt.create_from_ast(constraint_str)
    error_str = "The alternate reward number you entered was not an integer."
    assert str(excinfo.value) == error_str

    constraint_str = "J_pi_new_PDIS_[M] - 0.5"

    pt = ParseTree(delta, regime="reinforcement_learning", sub_regime="all")
    with pytest.raises(RuntimeError) as excinfo:
        pt.create_from_ast(constraint_str)
    error_str = "The alternate reward number you entered was not an integer."
    assert str(excinfo.value) == error_str

    constraint_str = "J_pi_new_PDIS_[5.9] - 0.5"

    pt = ParseTree(delta, regime="reinforcement_learning", sub_regime="all")
    with pytest.raises(RuntimeError) as excinfo:
        pt.create_from_ast(constraint_str)
    error_str = "The alternate reward number you entered was not an integer."
    assert str(excinfo.value) == error_str

    constraint_str = "J_pi_new_WIS_[M] - 0.5"

    pt = ParseTree(delta, regime="reinforcement_learning", sub_regime="all")
    with pytest.raises(RuntimeError) as excinfo:
        pt.create_from_ast(constraint_str)
    error_str = "The alternate reward number you entered was not an integer."
    assert str(excinfo.value) == error_str

    constraint_str = "J_pi_new_WIS_[5.9] - 0.5"

    pt = ParseTree(delta, regime="reinforcement_learning", sub_regime="all")
    with pytest.raises(RuntimeError) as excinfo:
        pt.create_from_ast(constraint_str)
    error_str = "The alternate reward number you entered was not an integer."
    assert str(excinfo.value) == error_str

def test_rl_alt_reward_precalc_return():
    np.random.seed(0)

    from seldonian.RL.RL_model import RL_model
    from seldonian.RL.Agents.Policies.Softmax import DiscreteSoftmax
    from seldonian.RL.Env_Description import Spaces, Env_Description

    data_pth = "static/datasets/RL/gridworld/gridworld_100episodes_2altrewards.pkl"
    loader = DataSetLoader(regime="reinforcement_learning")
    dataset = loader.load_RL_dataset_from_episode_file(data_pth)

    # Initialize policy
    num_states = 9
    observation_space = Spaces.Discrete_Space(0, num_states - 1)
    action_space = Spaces.Discrete_Space(0, 3)
    env_description = Env_Description.Env_Description(observation_space, action_space)
    policy = DiscreteSoftmax(
        hyperparam_and_setting_dict={}, env_description=env_description
    )
    env_kwargs = {"gamma": 0.9}
    model = RL_model(policy=policy, env_kwargs=env_kwargs)

    IS_constraint_strs = ["J_pi_new_IS_[1] >= -0.25"]
    IS_deltas = [0.05]

    IS_pt = ParseTree(
        delta=IS_deltas[0],
        regime="reinforcement_learning",
        sub_regime="all",
        columns=[],
    )

    # IS_pt.create_from_ast(IS_constraint_strs[0])
    # IS_pt.assign_deltas(weight_method="equal")
    IS_pt.build_tree(IS_constraint_strs[0])

    # propagate the bounds with example theta value
    theta = np.random.uniform(-0.05, 0.05, (9, 4))
    IS_pt.propagate_bounds(
        theta=theta,
        tree_dataset_dict={"all":dataset},
        model=model,
        branch="candidate_selection",
        regime="reinforcement_learning",
        n_safety=150,
        sub_regime="all"
    )
    assert IS_pt.root.upper == pytest.approx(7.29027)
    IS_pt.reset_base_node_dict(reset_data=True)
    IS_pt.evaluate_constraint(
        theta=theta,
        tree_dataset_dict={"all":dataset},
        model=model,
        regime="reinforcement_learning",
        branch="safety_test",
        sub_regime="all"
    )
    assert IS_pt.root.value == pytest.approx(3.5460865367462726)

    weighted_returns_alt_reward = IS_pt.base_node_dict["J_pi_new_IS_[1]"]["data_dict"][
        "weighted_returns"
    ]
    assert weighted_returns_alt_reward[0] == pytest.approx(7.563782445399999)

    PDIS_constraint_strs = ["J_pi_new_PDIS_[1] >= -0.25"]
    PDIS_deltas = [0.05]

    PDIS_pt = ParseTree(
        delta=PDIS_deltas[0],
        regime="reinforcement_learning",
        sub_regime="all",
        columns=[],
    )

    PDIS_pt.build_tree(PDIS_constraint_strs[0])

    # propagate the bounds with example theta value
    PDIS_pt.propagate_bounds(
        theta=theta,
        tree_dataset_dict={"all":dataset},
        model=model,
        branch="candidate_selection",
        regime="reinforcement_learning",
        n_safety=150,
        sub_regime="all"
    )
    assert PDIS_pt.root.upper == pytest.approx(0.2700371570087783)
    PDIS_pt.reset_base_node_dict(reset_data=True)
    PDIS_pt.evaluate_constraint(
        theta=theta,
        tree_dataset_dict={"all":dataset},
        model=model,
        regime="reinforcement_learning",
        branch="safety_test",
        sub_regime="all"
    )
    assert PDIS_pt.root.value == pytest.approx(0.08025886028)

    weighted_returns_alt_reward = PDIS_pt.base_node_dict["J_pi_new_PDIS_[1]"][
        "data_dict"
    ]["weighted_returns"]
    assert weighted_returns_alt_reward[0] == pytest.approx(7.563782445399999)

def test_measure_function_with_conditional_bad_syntax_captured():
    delta = 0.05
    error_str = (
        "Error parsing your expression."
        " The issue is most likely due to"
        " missing/mismatched parentheses or square brackets"
        " in a conditional expression involving '|'."
    )

    bad_constraint_strs = [
        "Mean_Error | M ",
        "(Mean_Error | M)",
        "(Mean_Error | M,F)",
        "abs(Mean_Error | [M] - Mean_Error | [F]) - 0.1",
        "abs((Mean_Error | M) - (Mean_Error | F)) - 0.1",
        "abs((Mean_Error | [M]) - (Mean_Error | F)) - 0.1",
        "abs((Mean_Error | M) - (Mean_Error | [F])) - 0.1",
        "abs((Mean_Error | [M]) - (Mean_Error | F,L)) - 0.1",
        "abs((Mean_Error | A,B) - (Mean_Error | [F])) - 0.1",
    ]

    for constraint_str in bad_constraint_strs:
        pt = ParseTree(
            delta,
            regime="supervised_learning",
            sub_regime="regression",
            columns=["A", "B", "M", "F", "L"],
        )
        with pytest.raises(RuntimeError) as excinfo:
            pt.create_from_ast(constraint_str)

        assert str(excinfo.value) == error_str

    constraint_str = "(Mean_Error | [G])"
    pt = ParseTree(delta, regime="supervised_learning", sub_regime="regression")
    with pytest.raises(RuntimeError) as excinfo:
        pt.create_from_ast(constraint_str)
    error_str = (
        "A column provided in your constraint str: G "
        "was not in the list of  columns provided: []"
    )
    assert str(excinfo.value) == error_str

def test_measure_function_from_wrong_regime():
    """Test that if a measure function from the incorrect
    regime or sub-regime is used in a constraint
    that the parse tree builder will raise an error"""
    delta = 0.05

    constraint_str = "Mean_Squared_Error - 2.0"

    pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")

    with pytest.raises(NotImplementedError) as excinfo:
        pt.create_from_ast(constraint_str)

    error_str = (
        "NotImplementedError: Error parsing your expression. "
        "A variable name was used which we do not recognize: "
        "Mean_Squared_Error"
    )
    assert str(excinfo.value) in error_str

    constraint_str = "FPR - 0.2"

    pt = ParseTree(delta, regime="supervised_learning", sub_regime="regression")

    with pytest.raises(NotImplementedError) as excinfo:
        pt.create_from_ast(constraint_str)

    error_str = (
        "NotImplementedError: Error parsing your expression. "
        "A variable name was used which we do not recognize: "
        "FPR"
    )
    assert str(excinfo.value) in error_str

    constraint_str = "Mean_Squared_Error - 2.0"

    pt = ParseTree(delta, regime="reinforcement_learning", sub_regime="all")

    with pytest.raises(NotImplementedError) as excinfo:
        pt.create_from_ast(constraint_str)

    error_str = (
        "NotImplementedError: Error parsing your expression. "
        "A variable name was used which we do not recognize: "
        "Mean_Squared_Error"
    )
    assert str(excinfo.value) in error_str

    constraint_str = "(FPR | [M]) - 0.2"

    pt = ParseTree(
        delta, regime="reinforcement_learning", sub_regime="all", columns=["M"]
    )

    with pytest.raises(NotImplementedError) as excinfo:
        pt.create_from_ast(constraint_str)

    error_str = (
        "NotImplementedError: Error parsing your expression. "
        "A variable name was used which we do not recognize: "
        "FPR"
    )
    assert str(excinfo.value) in error_str

def test_provided_measure_function_custom_regime():
    """Test that the functionality of providing a custom measure function 
    in the custom regime works as expected """

    # First test that one can provide a custom measure function
    delta = 0.05
    regime = "custom"
    sub_regime = None
    # Define behavioral constraint
    constraint_str = 'CUST_LOSS <= 30.0'
    delta = 0.05

    # Define custom measure function for CPR and register it when making parse tree
    def custom_measure_function(model, theta, data, **kwargs):
        """
        Calculate 
        for each observation. Meaning depends on whether
        binary or multi-class classification.

        :param model: SeldonianModel instance
        :param theta: The parameter weights
        :type theta: numpy ndarray
        :param data: A list of samples, where in this case samples are
            lists of length three with each element a single character

        :return: Positive rate for each observation
        :rtype: numpy ndarray(float between 0 and 1)
        """
        predictions = model.predict(theta,data)
        return predictions

    custom_measure_functions = {
        "CUST_LOSS": custom_measure_function
        }

    # Create parse tree object
    pt = ParseTree(
        delta=delta, regime=regime, sub_regime=sub_regime, columns=[],
        custom_measure_functions=custom_measure_functions
    )

    pt.build_tree(constraint_str)
    assert "CUST_LOSS" in pt.available_measure_functions
    assert "PR" not in pt.available_measure_functions
    assert "Mean_Squared_Error" not in pt.available_measure_functions

    # Now test that if one does not provide a custom measure function and tries to use it
    # an error is raised
    # Define behavioral constraint
    constraint_str = 'XYZ <= 0'
    delta = 0.05

    # Create parse tree object
    pt = ParseTree(
        delta=delta, regime=regime, sub_regime=sub_regime, columns=[]
    )

    with pytest.raises(NotImplementedError) as excinfo:
        pt.build_tree(constraint_str)
    
    error_str = (
        "Error parsing your expression. "
        "A variable name was used which we do not recognize: XYZ"
    )
    assert str(excinfo.value) == error_str


def test_custom_base_nodes():
    constraint_str = "MED_MF - 0.1"
    delta = 0.05

    pt = ParseTree(delta, regime="supervised_learning", sub_regime="regression")
    pt.create_from_ast(constraint_str)
    assert isinstance(pt.root.left, BaseNode)
    assert isinstance(pt.root.left, MEDCustomBaseNode)
    assert pt.n_base_nodes == 1
    assert len(pt.base_node_dict) == 1

    constraint_str = "CVaRSQE - 1.0"
    delta = 0.05

    pt = ParseTree(delta, regime="supervised_learning", sub_regime="regression")
    pt.create_from_ast(constraint_str)
    assert isinstance(pt.root.left, BaseNode)
    assert isinstance(pt.root.left, CVaRSQeBaseNode)
    assert pt.root.left.alpha == 0.1
    assert pt.n_base_nodes == 1
    assert len(pt.base_node_dict) == 1

def test_unary_op():
    delta = 0.05

    constraint_str = "-10+abs(Mean_Error)"
    pt = ParseTree(delta, regime="supervised_learning", sub_regime="regression")
    pt.create_from_ast(constraint_str)
    assert pt.root.name == "add"
    assert pt.root.left.value == -10
    assert pt.root.left.name == "-10"
    assert pt.root.right.name == "abs"
    assert pt.root.right.left.name == "Mean_Error"
    assert pt.n_nodes == 4
    assert pt.n_base_nodes == 1

    constraint_str = "-MED_MF"
    pt = ParseTree(delta, regime="supervised_learning", sub_regime="regression")
    pt.create_from_ast(constraint_str)
    assert pt.root.name == "mult"
    assert pt.root.left.value == -1
    assert pt.root.right.name == "MED_MF"
    assert pt.n_nodes == 3

    constraint_str = "-abs(Mean_Error) - 10"
    pt = ParseTree(delta, regime="supervised_learning", sub_regime="regression")
    pt.create_from_ast(constraint_str)
    assert pt.root.name == "sub"
    assert pt.root.right.value == 10
    assert pt.root.left.name == "mult"
    assert pt.root.left.left.value == -1
    assert pt.root.left.right.name == "abs"
    assert pt.root.left.right.left.name == "Mean_Error"
    assert pt.n_nodes == 6
    assert pt.n_base_nodes == 1
    assert len(pt.base_node_dict) == 1

    constraint_str = "-abs(Mean_Error | [M]) - 10"
    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="regression", columns=["M"]
    )
    pt.create_from_ast(constraint_str)
    assert pt.root.name == "sub"
    assert pt.root.right.value == 10
    assert pt.root.left.name == "mult"
    assert pt.root.left.left.value == -1
    assert pt.root.left.right.name == "abs"
    assert pt.root.left.right.left.name == "Mean_Error | [M]"
    assert pt.n_nodes == 6
    assert pt.n_base_nodes == 1
    assert len(pt.base_node_dict) == 1

    constraint_str = "+Mean_Error"
    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="regression", columns=["M"]
    )
    with pytest.raises(NotImplementedError) as excinfo:
        pt.create_from_ast(constraint_str)
    error_str = (
        "Error parsing your expression."
        " A unary operator was used which we do not support: "
        f"+"
    )
    assert str(excinfo.value) == error_str

def test_raise_error_on_excluded_operators():
    constraint_str = "FPR^4"
    delta = 0.05
    pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")
    with pytest.raises(NotImplementedError) as excinfo:
        pt.create_from_ast(constraint_str)
    error_str = (
        "Error parsing your expression."
        " An operator was used which we do not support: ^"
    )
    assert str(excinfo.value) == error_str

    constraint_str = "FPR<4"
    delta = 0.05
    pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")
    with pytest.raises(NotImplementedError) as excinfo:
        pt.create_from_ast(constraint_str)
    error_str = (
        "Error parsing your expression."
        " An operator was used which we do not support: <"
    )
    assert str(excinfo.value) == error_str

    constraint_str = "FPR>4"
    delta = 0.05
    pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")
    with pytest.raises(NotImplementedError) as excinfo:
        pt.create_from_ast(constraint_str)
    error_str = (
        "Error parsing your expression."
        " An operator was used which we do not support: >"
    )
    assert str(excinfo.value) == error_str

    constraint_str = "FPR & FNR"
    delta = 0.05
    pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")
    with pytest.raises(NotImplementedError) as excinfo:
        pt.create_from_ast(constraint_str)
    error_str = (
        "Error parsing your expression."
        " An operator was used which we do not support: &"
    )
    assert str(excinfo.value) == error_str

    constraint_str = "FPR//4"
    delta = 0.05
    pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")
    with pytest.raises(NotImplementedError) as excinfo:
        pt.create_from_ast(constraint_str)
    error_str = (
        "Error parsing your expression."
        " An operator was used which we do not support: //"
    )
    assert str(excinfo.value) == error_str

def test_single_conditional_columns_assigned():
    constraint_str = "abs(Mean_Error|[X]) - 0.1"
    delta = 0.05
    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="regression", columns=["X"]
    )
    pt.create_from_ast(constraint_str)
    assert pt.n_nodes == 4
    assert pt.n_base_nodes == 1
    assert len(pt.base_node_dict) == 1
    assert pt.root.left.left.conditional_columns == ["X"]

def test_multiple_conditional_columns_assigned():
    constraint_str = "abs(Mean_Error|[X,Y,Z]) - 0.1"
    delta = 0.05
    pt = ParseTree(
        delta,
        regime="supervised_learning",
        sub_regime="regression",
        columns=["X", "Y", "Z"],
    )
    pt.create_from_ast(constraint_str)
    assert pt.n_nodes == 4
    assert pt.n_base_nodes == 1
    assert len(pt.base_node_dict) == 1
    assert pt.root.left.left.conditional_columns == ["X", "Y", "Z"]

def test_math_functions_propagate():
    np.random.seed(0)
    data_pth = "static/datasets/supervised/GPA/gpa_classification_dataset.csv"
    metadata_pth = "static/datasets/supervised/GPA/metadata_classification.json"

    metadata_dict = load_json(metadata_pth)

    loader = DataSetLoader(regime="supervised_learning")

    dataset = loader.load_supervised_dataset(
        filename=data_pth, metadata_filename=metadata_pth, file_type="csv"
    )

    model_instance = LinearRegressionModel()

    constraint_str = "exp(FPR - 0.5)"
    delta = 0.05
    pt = ParseTree(
        delta,
        regime="supervised_learning",
        sub_regime="classification",
        columns=dataset.meta.sensitive_col_names,
    )

    pt.create_from_ast(constraint_str)
    pt.assign_deltas(weight_method="equal")

    # propagate the bounds with example theta value
    # theta = np.hstack([np.array([0.0,0.0]),np.random.uniform(-0.05,0.05,10)])
    theta = np.random.uniform(-0.05, 0.05, 10)
    pt.propagate_bounds(
        theta=theta,
        tree_dataset_dict={"all":dataset},
        model=model_instance,
        branch="safety_test",
        regime="supervised_learning",
        sub_regime="classification"
    )
    assert pt.root.lower == pytest.approx(0.59886236)
    assert pt.root.upper == pytest.approx(0.60010258)

    # constraint_str = '1+log(FPR)'
    # delta = 0.05
    # pt = ParseTree(delta,regime='supervised_learning',
    # 	sub_regime='classification',
    # 	columns=dataset.meta.sensitive_col_names)
    # # pt.build_tree(constraint_str)

    # pt.create_from_ast(constraint_str)
    # pt.assign_deltas(weight_method='equal')

    # # propagate the bounds with example theta value
    # theta = np.random.uniform(-0.05,0.05,10)
    # pt.propagate_bounds(theta=theta,dataset=dataset,
    # 	model=model_instance,branch='safety_test',
    # 	regime='supervised_learning')
    # assert pt.root.lower == pytest.approx(-2.5187904943528903)
    # assert pt.root.upper == pytest.approx(-2.470509955060809)

def test_deltas_assigned_equally():
    constraint_str = "abs((Mean_Error|[M]) - (Mean_Error|[F])) - 0.1"
    delta = 0.05

    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="regression", columns=["M", "F"]
    )
    pt.create_from_ast(constraint_str)
    pt.assign_bounds_needed()
    pt.assign_deltas(weight_method="equal")
    assert pt.n_nodes == 6
    assert pt.n_base_nodes == 2
    n_unique_base_nodes = len(pt.base_node_dict)
    assert n_unique_base_nodes == 2
    assert isinstance(pt.root, InternalNode)
    assert pt.root.name == "sub"
    assert pt.root.left.left.left.delta_lower == delta / (2 * n_unique_base_nodes)
    assert pt.root.left.left.right.delta_upper == delta / (2 * n_unique_base_nodes)

def test_deltas_assigned_once_per_unique_basenode():
    """Make sure that the delta assigned to each base node
    is delta/number_of_unique_base_nodes, such that if a base
    node appears more than once it doesn't further dilute delta
    """
    constraint_str = "0.8 - min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M]))"
    delta = 0.05

    pt = ParseTree(
        delta,
        regime="supervised_learning",
        sub_regime="classification",
        columns=["M", "F"],
    )
    pt.create_from_ast(constraint_str)
    pt.assign_bounds_needed()
    pt.assign_deltas(weight_method="equal")
    assert pt.n_nodes == 9
    assert pt.n_base_nodes == 4
    n_unique_base_nodes = len(pt.base_node_dict)
    assert n_unique_base_nodes == 2
    # assert pt.root.name == 'sub'
    # print(pt.base_node_dict)
    assert pt.root.right.left.left.delta_lower == delta / (2 * n_unique_base_nodes)
    assert pt.root.right.left.left.delta_upper == delta / (2 * n_unique_base_nodes)
    assert pt.root.right.left.right.delta_lower == delta / (2 * n_unique_base_nodes)
    assert pt.root.right.left.right.delta_upper == delta / (2 * n_unique_base_nodes)
    assert pt.root.right.right.left.delta_lower == delta / (2 * n_unique_base_nodes)
    assert pt.root.right.right.left.delta_upper == delta / (2 * n_unique_base_nodes)
    assert pt.root.right.right.right.delta_lower == delta / (2 * n_unique_base_nodes)
    assert pt.root.right.right.right.delta_upper == delta / (2 * n_unique_base_nodes)

def test_deltas_assigned_correctly_not_all_bounds_needed():
    """Make sure that when not all bounds are needed,
    the delta values that get assigned to each base node bound
    reflect that.
    """
    delta = 0.05  # use for all trees below

    constraint_str = "FPR"
    pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")
    pt.create_from_ast(constraint_str)
    # Before bounds assigned both should be True
    assert pt.root.will_lower_bound == True
    assert pt.root.will_upper_bound == True
    pt.assign_bounds_needed()
    pt.assign_deltas(weight_method="equal")
    # Now, we should find that only upper bound is needed

    assert pt.root.will_lower_bound == False
    assert pt.root.will_upper_bound == True

    # This means that the upper bound should get all of the delta and delta_lower should be 0
    assert pt.root.delta_lower == None
    assert pt.root.delta_upper == delta

    constraint_str = "(Mean_Error | [M]) - 0.1"
    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="regression", columns=["M"]
    )
    pt.create_from_ast(constraint_str)
    # Before bounds assigned both should be True
    assert pt.root.left.name == "Mean_Error | [M]"
    assert pt.root.left.will_lower_bound == True
    assert pt.root.left.will_upper_bound == True
    pt.assign_bounds_needed()
    pt.assign_deltas(weight_method="equal")
    # But after, we should find that only upper is needed
    assert pt.n_nodes == 3
    assert pt.n_base_nodes == 1
    assert isinstance(pt.root.left, BaseNode)
    assert pt.root.left.will_lower_bound == False
    assert pt.root.left.will_upper_bound == True

    # This means that the upper bound should get all of the delta and delta_lower should be 0
    assert pt.root.left.delta_lower == None
    assert pt.root.left.delta_upper == delta

    constraint_str = "2.0 - Mean_Squared_Error"
    pt = ParseTree(delta, regime="supervised_learning", sub_regime="regression")
    pt.create_from_ast(constraint_str)
    # Before bounds assigned both should be True
    assert pt.root.right.name == "Mean_Squared_Error"
    assert pt.root.right.will_lower_bound == True
    assert pt.root.right.will_upper_bound == True
    pt.assign_bounds_needed()
    pt.assign_deltas(weight_method="equal")

    # But after, we should find that only lower is needed
    assert pt.n_nodes == 3
    assert pt.n_base_nodes == 1
    assert isinstance(pt.root.right, BaseNode)
    assert pt.root.right.will_lower_bound == True
    assert pt.root.right.will_upper_bound == False

    # This means that the upper bound should get all of the delta and delta_lower should be 0
    assert pt.root.right.delta_lower == delta
    assert pt.root.right.delta_upper == None

    constraint_str = "(FPR * FNR) - 0.25"
    pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")
    pt.create_from_ast(constraint_str)
    # Before bounds assigned both should be True
    assert pt.root.left.left.name == "FPR"
    assert pt.root.left.right.name == "FNR"

    assert pt.root.left.left.will_lower_bound == True
    assert pt.root.left.left.will_upper_bound == True
    assert pt.root.left.right.will_lower_bound == True
    assert pt.root.left.right.will_upper_bound == True
    pt.assign_bounds_needed()
    pt.assign_deltas(weight_method="equal")
    # After, we should find that both base nodes need both still
    assert pt.n_nodes == 5
    assert pt.n_base_nodes == 2
    assert pt.root.left.left.will_lower_bound == True
    assert pt.root.left.left.will_upper_bound == True
    assert pt.root.left.right.will_lower_bound == True
    assert pt.root.left.right.will_upper_bound == True

    # This means that the upper and lower bounds of each node should get 1/4 of the total delta
    assert pt.root.left.left.delta_lower == delta / 4
    assert pt.root.left.left.delta_upper == delta / 4
    assert pt.root.left.right.delta_lower == delta / 4
    assert pt.root.left.right.delta_upper == delta / 4

def test_delta_vector_assignment():
    # Test case #1: FPR needing only an upper bound with custom delta = pt.delta
    tree_delta = 0.05
    constraint_str = "FPR"
    pt = ParseTree(
        tree_delta, regime="supervised_learning", sub_regime="classification"
    )
    pt.build_tree(constraint_str,delta_weight_method="manual", delta_vector=[0.05])
    
    # Should find that only the upper bound on FPR is needed
    # and upper bound gets assigned the correct delta
    assert pt.root.delta_lower == None
    assert pt.root.delta_upper == 0.05
    assert pt.base_node_dict["FPR"]["delta_lower"] == None
    assert pt.base_node_dict["FPR"]["delta_upper"] == 0.05

    # Test case #2: FPR needing only an upper bound with custom delta != pt.delta
    # In this case, custom delta should get renormalized to pt.delta
    tree_delta = 0.05
    constraint_str = "FPR"
    pt = ParseTree(
        tree_delta, regime="supervised_learning", sub_regime="classification"
    )
    pt.build_tree(constraint_str,delta_weight_method="manual", delta_vector=[0.06])
    assert pt.root.delta_lower == None
    assert pt.root.delta_upper == 0.05
    assert pt.base_node_dict["FPR"]["delta_lower"] == None
    assert pt.base_node_dict["FPR"]["delta_upper"] == 0.05

    # Test case #3: two base nodes each needing both bounds.
    # Providing a delta vector with different values but correctly
    # normalized
    tree_delta = 0.05
    constraint_str = "abs((Mean_Error | [M]) - (Mean_Error | [F])) - 0.1"
    pt = ParseTree(
        tree_delta,
        regime="supervised_learning",
        sub_regime="regression",
        columns=["M", "F"],
    )
    pt.build_tree(constraint_str,delta_weight_method="manual", delta_vector=[0.01, 0.02, 0.03, 0.04])
    reweighted_deltas = [
        0.012313129661287526,
        0.012436878671712491,
        0.01256187138036873,
        0.012688120286631263,
    ]
    assert pt.root.left.left.left.delta_lower == reweighted_deltas[0]
    assert pt.root.left.left.left.delta_upper == reweighted_deltas[1]
    assert pt.root.left.left.right.delta_lower == reweighted_deltas[2]
    assert pt.root.left.left.right.delta_upper == reweighted_deltas[3]
    assert pt.base_node_dict["Mean_Error | [M]"]["delta_lower"] == reweighted_deltas[0]
    assert pt.base_node_dict["Mean_Error | [M]"]["delta_upper"] == reweighted_deltas[1]
    assert pt.base_node_dict["Mean_Error | [F]"]["delta_lower"] == reweighted_deltas[2]
    assert pt.base_node_dict["Mean_Error | [F]"]["delta_upper"] == reweighted_deltas[3]

    # Test case #4: two base nodes each needing both bounds.
    # Providing a delta vector with different values and not correctly
    # normalized
    tree_delta = 0.1
    constraint_str = "abs((Mean_Error | [M]) - (Mean_Error | [F])) - 0.1"
    pt = ParseTree(
        tree_delta,
        regime="supervised_learning",
        sub_regime="regression",
        columns=["M", "F"],
    )
    pt.build_tree(constraint_str, delta_weight_method="manual", delta_vector=[-1, -0.2, 0, 2])
    reweighted_deltas = [
        0.0038418155970352825,
        0.008550117850922747,
        0.010443137525711545,
        0.07716492902633044,
    ]
    assert pt.root.left.left.left.delta_lower == reweighted_deltas[0]
    assert pt.root.left.left.left.delta_upper == reweighted_deltas[1]
    assert pt.root.left.left.right.delta_lower == reweighted_deltas[2]
    assert pt.root.left.left.right.delta_upper == reweighted_deltas[3]
    assert pt.base_node_dict["Mean_Error | [M]"]["delta_lower"] == reweighted_deltas[0]
    assert pt.base_node_dict["Mean_Error | [M]"]["delta_upper"] == reweighted_deltas[1]
    assert pt.base_node_dict["Mean_Error | [F]"]["delta_lower"] == reweighted_deltas[2]
    assert pt.base_node_dict["Mean_Error | [F]"]["delta_upper"] == reweighted_deltas[3]

    # Test case #5: two base nodes each needing both bounds.
    # Providing a delta vector as a numpy array is OK as long
    # as it is 1D.
    tree_delta = 0.05
    constraint_str = "abs((Mean_Error | [M]) - (Mean_Error | [F])) - 0.1"
    pt = ParseTree(
        tree_delta,
        regime="supervised_learning",
        sub_regime="regression",
        columns=["M", "F"],
    )
    pt.build_tree(constraint_str,
        delta_weight_method="manual", delta_vector=np.array([0.01, 0.02, 0.03, 0.04])
    )
    reweighted_deltas = [
        0.012313129661287526,
        0.012436878671712491,
        0.01256187138036873,
        0.012688120286631263,
    ]
    assert pt.root.left.left.left.delta_lower == reweighted_deltas[0]
    assert pt.root.left.left.left.delta_upper == reweighted_deltas[1]
    assert pt.root.left.left.right.delta_lower == reweighted_deltas[2]
    assert pt.root.left.left.right.delta_upper == reweighted_deltas[3]
    assert pt.base_node_dict["Mean_Error | [M]"]["delta_lower"] == reweighted_deltas[0]
    assert pt.base_node_dict["Mean_Error | [M]"]["delta_upper"] == reweighted_deltas[1]
    assert pt.base_node_dict["Mean_Error | [F]"]["delta_lower"] == reweighted_deltas[2]
    assert pt.base_node_dict["Mean_Error | [F]"]["delta_upper"] == reweighted_deltas[3]

    # Test case #6: Providing a delta vector of the wrong length raises a ValueError
    tree_delta = 0.1
    constraint_str = "abs((Mean_Error | [M]) - (Mean_Error | [F])) - 0.1"
    pt = ParseTree(
        tree_delta,
        regime="supervised_learning",
        sub_regime="regression",
        columns=["M", "F"],
    )

    with pytest.raises(ValueError) as excinfo:
        pt.build_tree(constraint_str,delta_weight_method="manual", delta_vector=[0.01, 0.02, 0.03])

    error_str = "delta_vector has length: 3, but should be of length: 4"
    assert str(excinfo.value) == error_str

    # Test case #7: Providing a delta vector of the wrong data type
    tree_delta = 0.1
    constraint_str = "abs((Mean_Error | [M]) - (Mean_Error | [F])) - 0.1"
    pt = ParseTree(
        tree_delta,
        regime="supervised_learning",
        sub_regime="regression",
        columns=["M", "F"],
    )
    with pytest.raises(ValueError) as excinfo:
        pt.build_tree(constraint_str,delta_weight_method="manual", delta_vector=(0.01, 0.02, 0.03))

    error_str = "delta_vector must be a list or 1D numpy array"
    assert str(excinfo.value) == error_str

    # Test case #8: Providing a delta vector as a numpy array of the wrong shape
    tree_delta = 0.1
    constraint_str = "abs((Mean_Error | [M]) - (Mean_Error | [F])) - 0.1"
    pt = ParseTree(
        tree_delta,
        regime="supervised_learning",
        sub_regime="regression",
        columns=["M", "F"],
    )
    with pytest.raises(ValueError) as excinfo:
        pt.build_tree(constraint_str,
            delta_weight_method="manual", delta_vector=np.array([[0.01], [0.02], [0.03]])
        )

    error_str = "delta_vector must be a list or 1D numpy array"
    assert str(excinfo.value) == error_str

    # Test case #9: Trying to assign deltas before assinging bounds raises a RuntimeError
    tree_delta = 0.1
    constraint_str = "abs((Mean_Error | [M]) - (Mean_Error | [F])) - 0.1"
    pt = ParseTree(
        tree_delta,
        regime="supervised_learning",
        sub_regime="regression",
        columns=["M", "F"],
    )
    pt.create_from_ast(constraint_str)
    with pytest.raises(RuntimeError) as excinfo:
        pt.assign_deltas(weight_method="manual", delta_vector=[0.01, 0.1, 0.2, 0.3])

    error_str = (
        "Need to run assign_bounds_needed() before "
        "assigning deltas with a custom delta vector."
    )
    assert str(excinfo.value) == error_str

    # Test case #10: If softmaxing explodes due to e^(large), raise ValueError
    tree_delta = 0.1
    constraint_str = "abs((Mean_Error | [M]) - (Mean_Error | [F])) - 0.1"
    pt = ParseTree(
        tree_delta,
        regime="supervised_learning",
        sub_regime="regression",
        columns=["M", "F"],
    )
    delta_vector = [1000, 0.1, 0.2, 0.3]
    with pytest.raises(ValueError) as excinfo:
        pt.build_tree(constraint_str,delta_weight_method="manual", delta_vector=delta_vector)

    error_str = f"softmaxing delta_vector={delta_vector} resulted in nan or inf."
    assert str(excinfo.value) == error_str

def test_bounds_needed_assigned_correctly():
    delta = 0.05  # use for all trees below

    constraint_str = "FPR"
    pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")
    assert pt.n_nodes == 0
    pt.create_from_ast(constraint_str)
    # Before bounds assigned both should be True
    assert pt.root.will_lower_bound == True
    assert pt.root.will_upper_bound == True
    assert pt.n_unique_bounds_tot is None
    pt.assign_bounds_needed()
    # But after, we should find that only upper is needed
    assert pt.n_nodes == 1
    assert pt.n_base_nodes == 1
    assert isinstance(pt.root, BaseNode)
    assert pt.root.will_lower_bound == False
    assert pt.root.will_upper_bound == True
    assert pt.n_unique_bounds_tot == 1

    constraint_str = "(Mean_Error | [M]) - 0.1"
    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="regression", columns=["M"]
    )
    pt.create_from_ast(constraint_str)
    # Before bounds assigned both should be True
    assert pt.root.left.name == "Mean_Error | [M]"
    assert pt.root.left.will_lower_bound == True
    assert pt.root.left.will_upper_bound == True
    assert pt.n_unique_bounds_tot is None
    pt.assign_bounds_needed()
    # But after, we should find that only upper is needed
    assert pt.n_nodes == 3
    assert pt.n_base_nodes == 1
    assert isinstance(pt.root.left, BaseNode)
    assert pt.root.left.will_lower_bound == False
    assert pt.root.left.will_upper_bound == True
    assert pt.n_unique_bounds_tot == 1

    constraint_str = "2.0 - Mean_Squared_Error"
    pt = ParseTree(delta, regime="supervised_learning", sub_regime="regression")
    pt.create_from_ast(constraint_str)
    # Before bounds assigned both should be True
    assert pt.root.right.name == "Mean_Squared_Error"
    assert pt.root.right.will_lower_bound == True
    assert pt.root.right.will_upper_bound == True
    assert pt.n_unique_bounds_tot is None
    pt.assign_bounds_needed()
    # But after, we should find that only lower is needed
    assert pt.n_nodes == 3
    assert pt.n_base_nodes == 1
    assert isinstance(pt.root.right, BaseNode)
    assert pt.root.right.will_lower_bound == True
    assert pt.root.right.will_upper_bound == False
    assert pt.n_unique_bounds_tot == 1

    constraint_str = "abs((Mean_Error | [M]) - (Mean_Error | [F])) - 0.1"
    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="regression", columns=["M", "F"]
    )
    pt.create_from_ast(constraint_str)
    # Before bounds assigned both should be True
    assert pt.root.left.left.left.name == "Mean_Error | [M]"
    assert pt.root.left.left.left.will_lower_bound == True
    assert pt.root.left.left.left.will_upper_bound == True
    assert pt.root.left.left.right.name == "Mean_Error | [F]"
    assert pt.root.left.left.right.will_lower_bound == True
    assert pt.root.left.left.right.will_upper_bound == True
    assert pt.n_unique_bounds_tot is None
    pt.assign_bounds_needed()
    # # After, we should find that both base nodes need both still
    assert pt.n_nodes == 6
    assert pt.n_base_nodes == 2
    assert pt.root.left.left.left.will_lower_bound == True
    assert pt.root.left.left.left.will_upper_bound == True
    assert pt.root.left.left.right.will_lower_bound == True
    assert pt.root.left.left.right.will_upper_bound == True
    assert pt.n_unique_bounds_tot == 4

    constraint_str = "(FPR * FNR) - 0.25"
    pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")
    pt.create_from_ast(constraint_str)
    # Before bounds assigned both should be True
    assert pt.root.left.left.name == "FPR"
    assert pt.root.left.right.name == "FNR"

    assert pt.root.left.left.will_lower_bound == True
    assert pt.root.left.left.will_upper_bound == True
    assert pt.root.left.right.will_lower_bound == True
    assert pt.root.left.right.will_upper_bound == True
    assert pt.n_unique_bounds_tot is None
    pt.assign_bounds_needed()
    # After, we should find that both base nodes need both still
    assert pt.n_nodes == 5
    assert pt.n_base_nodes == 2
    assert pt.root.left.left.will_lower_bound == True
    assert pt.root.left.left.will_upper_bound == True
    assert pt.root.left.right.will_lower_bound == True
    assert pt.root.left.right.will_upper_bound == True
    assert pt.n_unique_bounds_tot == 4

    constraint_str = "(TPR - FPR) - 0.25"
    pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")
    pt.create_from_ast(constraint_str)
    # Before bounds assigned both should be True
    assert pt.root.left.left.name == "TPR"
    assert pt.root.left.right.name == "FPR"

    assert pt.root.left.left.will_lower_bound == True
    assert pt.root.left.left.will_upper_bound == True
    assert pt.root.left.right.will_lower_bound == True
    assert pt.root.left.right.will_upper_bound == True
    assert pt.n_unique_bounds_tot is None
    pt.assign_bounds_needed()
    # After, we should find that TPR only needs upper and FPR only needs lower
    assert pt.n_nodes == 5
    assert pt.n_base_nodes == 2
    assert pt.root.left.left.will_lower_bound == False
    assert pt.root.left.left.will_upper_bound == True
    assert pt.root.left.right.will_lower_bound == True
    assert pt.root.left.right.will_upper_bound == False
    assert pt.n_unique_bounds_tot == 2

    constraint_str = "(FPR + FNR) - 0.25"
    pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")
    pt.create_from_ast(constraint_str)
    # Before bounds assigned both should be True
    assert pt.root.left.left.name == "FPR"
    assert pt.root.left.right.name == "FNR"

    assert pt.root.left.left.will_lower_bound == True
    assert pt.root.left.left.will_upper_bound == True
    assert pt.root.left.right.will_lower_bound == True
    assert pt.root.left.right.will_upper_bound == True
    assert pt.n_unique_bounds_tot is None
    pt.assign_bounds_needed()
    # After, we should find that both only need upper bound
    assert pt.n_nodes == 5
    assert pt.n_base_nodes == 2
    assert pt.root.left.left.will_lower_bound == False
    assert pt.root.left.left.will_upper_bound == True
    assert pt.root.left.right.will_lower_bound == False
    assert pt.root.left.right.will_upper_bound == True
    assert pt.n_unique_bounds_tot == 2

def test_bound_inflation_factor_assignment():
    delta = 0.05 # used for all test cases

    # Test case #1: Single base node with constant inflation factor (default=2)
    constraint_str = "FPR"
    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="classification"
    )
    pt.build_tree(constraint_str=constraint_str)
    
    assert pt.root.infl_factor_lower == None
    assert pt.root.infl_factor_upper == 2
    assert pt.base_node_dict["FPR"]["infl_factor_lower"] == None
    assert pt.base_node_dict["FPR"]["infl_factor_upper"] == 2

    # Test case #2: Single base node with constant inflation factor (non-default=1)
    infl_factor = 1
    constraint_str = "FPR"
    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="classification"
    )
    pt.build_tree(constraint_str=constraint_str,infl_factors=infl_factor)
    
    assert pt.root.infl_factor_lower == None
    assert pt.root.infl_factor_upper == 1
    assert pt.base_node_dict["FPR"]["infl_factor_lower"] == None
    assert pt.base_node_dict["FPR"]["infl_factor_upper"] == 1

    # Test case #3: Two base nodes with constant inflation factor (default=2)
    constraint_str = "abs((PR | [M]) - (PR | [F])) - 0.1"
    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="classification",
        columns=["M","F"]
    )
    pt.build_tree(constraint_str=constraint_str)
    
    assert pt.root.left.left.left.infl_factor_lower == 2
    assert pt.root.left.left.left.infl_factor_upper == 2
    assert pt.root.left.left.right.infl_factor_lower == 2
    assert pt.root.left.left.right.infl_factor_upper == 2

    assert pt.base_node_dict["PR | [M]"]["infl_factor_lower"] == 2
    assert pt.base_node_dict["PR | [M]"]["infl_factor_upper"] == 2
    assert pt.base_node_dict["PR | [F]"]["infl_factor_lower"] == 2
    assert pt.base_node_dict["PR | [F]"]["infl_factor_upper"] == 2

    # Test case #4: Two base nodes with constant non-default factors 
    infl_factor = 3
    constraint_str = "abs((PR | [M]) - (PR | [F])) - 0.1"
    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="classification",
        columns=["M","F"]
    )
    pt.build_tree(constraint_str=constraint_str,infl_factors=infl_factor)
    
    assert pt.root.left.left.left.infl_factor_lower == infl_factor
    assert pt.root.left.left.left.infl_factor_upper == infl_factor
    assert pt.root.left.left.right.infl_factor_lower == infl_factor
    assert pt.root.left.left.right.infl_factor_upper == infl_factor

    assert pt.base_node_dict["PR | [M]"]["infl_factor_lower"] == infl_factor
    assert pt.base_node_dict["PR | [M]"]["infl_factor_upper"] == infl_factor
    assert pt.base_node_dict["PR | [F]"]["infl_factor_lower"] == infl_factor
    assert pt.base_node_dict["PR | [F]"]["infl_factor_upper"] == infl_factor
    
    # Test case #5: Two base nodes with vector of factors 
    factors = [1.9,2.3,0.4,3]
    constraint_str = "abs((PR | [M]) - (PR | [F])) - 0.1"
    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="classification",
        columns=["M","F"]
    )
    pt.build_tree(constraint_str=constraint_str,infl_factor_method="manual",infl_factors=factors)
    
    assert pt.root.left.left.left.infl_factor_lower == factors[0]
    assert pt.root.left.left.left.infl_factor_upper == factors[1]
    assert pt.root.left.left.right.infl_factor_lower == factors[2]
    assert pt.root.left.left.right.infl_factor_upper == factors[3]

    assert pt.base_node_dict["PR | [M]"]["infl_factor_lower"] == factors[0]
    assert pt.base_node_dict["PR | [M]"]["infl_factor_upper"] == factors[1]
    assert pt.base_node_dict["PR | [F]"]["infl_factor_lower"] == factors[2]
    assert pt.base_node_dict["PR | [F]"]["infl_factor_upper"] == factors[3]

    # Test case #6: Four base nodes but only two unique with vector of factors 
    factors = [1.9,2.3,0.4,3]
    constraint_str = "min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M])) >= 0.8"
    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="classification",
        columns=["M","F"]
    )
    pt.build_tree(constraint_str=constraint_str,infl_factor_method="manual",infl_factors=factors)
    assert pt.root.right.left.left.infl_factor_lower == factors[0] # PR | [M] lower
    assert pt.root.right.left.left.infl_factor_upper == factors[1] # PR | [M] upper
    assert pt.root.right.left.right.infl_factor_lower == factors[2] # PR | [F] lower
    assert pt.root.right.left.right.infl_factor_upper == factors[3] # PR | [F] upper

    assert pt.root.right.right.left.infl_factor_lower == factors[2] # PR | [F] lower
    assert pt.root.right.right.left.infl_factor_upper == factors[3] # PR | [F] upper
    assert pt.root.right.right.right.infl_factor_lower == factors[0] # PR | [M] lower
    assert pt.root.right.right.right.infl_factor_upper == factors[1] # PR | [M] upper

    assert pt.base_node_dict["PR | [M]"]["infl_factor_lower"] == factors[0]
    assert pt.base_node_dict["PR | [M]"]["infl_factor_upper"] == factors[1]
    assert pt.base_node_dict["PR | [F]"]["infl_factor_lower"] == factors[2]
    assert pt.base_node_dict["PR | [F]"]["infl_factor_upper"] == factors[3]

    # Test case #7: Trying to run assign_infl_factors before assigning bounds results in error
    constraint_str = "FPR"
    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="classification"
    )
    pt.create_from_ast(constraint_str)
    with pytest.raises(RuntimeError) as excinfo:
        pt.assign_infl_factors()
    error_str = (
        "Need to run assign_bounds_needed() before "
        "assigning inflation factors."
    )
    assert str(excinfo.value) == error_str

    # Test case #8: Using "constant" method with a vector of factors results in error
    constraint_str = "FPR"
    factors = [1,2]
    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="classification"
    )
    with pytest.raises(ValueError) as excinfo:
        pt.build_tree(constraint_str=constraint_str,infl_factors=factors)
    
    error_str = (
        f"When method='constant', factors must be a single number"     
    )
    assert str(excinfo.value) == error_str

    # Test case #9: Using "manual" method with a single number results in error
    constraint_str = "FPR - FNR"
    infl_factor = 2.0
    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="classification"
    )
    with pytest.raises(ValueError) as excinfo:
        pt.build_tree(constraint_str=constraint_str,infl_factor_method="manual",infl_factors=infl_factor)
    
    error_str = (
        f"When method='manual', factors must be a list or 1D numpy array"     
    )
    assert str(excinfo.value) == error_str

    # Test case #10: factors list must be of correct length
    constraint_str = "FPR - FNR"
    factors = [2.0]
    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="classification"
    )
    with pytest.raises(ValueError) as excinfo:
        pt.build_tree(constraint_str=constraint_str,infl_factor_method="manual",infl_factors=factors)
    
    error_str = (
        f"factors has length: 1, but should be of length: 2"
    )
    assert str(excinfo.value) == error_str

    # Test case #11: factors must be positive
    constraint_str = "FPR - FNR"
    factors = [2.0,-0.8]
    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="classification"
    )
    with pytest.raises(ValueError) as excinfo:
        pt.build_tree(constraint_str=constraint_str,infl_factor_method="manual",infl_factors=factors)
    
    error_str = (
        f"factors must all be non-negative"
    )
    assert str(excinfo.value) == error_str

    # Test case #12: factors must be positive
    constraint_str = "FPR"
    factors = -1.4
    pt = ParseTree(
        delta, regime="supervised_learning", sub_regime="classification"
    )
    with pytest.raises(ValueError) as excinfo:
        pt.build_tree(constraint_str=constraint_str,infl_factors=factors)
    
    error_str = (
        f"factors must all be non-negative"
    )
    assert str(excinfo.value) == error_str

def test_duplicate_base_nodes():
    constraint_str = "FPR + 4/FPR - 2.0"
    delta = 0.05

    pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")
    pt.create_from_ast(constraint_str)
    assert pt.n_base_nodes == 2
    assert len(pt.base_node_dict) == 1
    assert pt.base_node_dict["FPR"]["bound_computed"] == False
    pt.base_node_dict["FPR"]["bound_method"] = "random"
    pt.propagate_bounds()
    assert pt.base_node_dict["FPR"]["bound_computed"] == True

def test_ttest_bound(simulated_regression_dataset):
    # dummy data for linear regression

    # First, single sided bound (MSE only needs upper bound)
    constraint_strs = ["Mean_Squared_Error - 2.0"]
    deltas = [0.05]
    frac_data_in_safety = 0.6

    (dataset, model, primary_objective, parse_trees) = simulated_regression_dataset(
        constraint_strs, deltas
    )

    features = dataset.features
    labels = dataset.labels

    (
        candidate_features,
        safety_features,
        candidate_labels,
        safety_labels,
    ) = train_test_split(features, labels, test_size=frac_data_in_safety, shuffle=False)

    candidate_dataset = SupervisedDataSet(
        features=candidate_features,
        labels=candidate_labels,
        sensitive_attrs=[],
        num_datapoints=len(candidate_features),
        meta=dataset.meta,
    )

    safety_dataset = SupervisedDataSet(
        features=safety_features,
        labels=safety_labels,
        sensitive_attrs=[],
        num_datapoints=len(safety_features),
        meta=dataset.meta,
    )


    pt = ParseTree(deltas[0], regime="supervised_learning", sub_regime="regression")
    pt.build_tree(constraint_strs[0])
    # pt.create_from_ast(constraint_strs[0])
    # pt.assign_bounds_needed()
    # pt.assign_deltas(weight_method="equal")
    # pt.assign_infl_factors()

    assert pt.n_nodes == 3
    assert pt.n_base_nodes == 1
    assert len(pt.base_node_dict) == 1
    assert pt.root.name == "sub"
    assert pt.root.left.will_lower_bound == False
    assert pt.root.left.will_upper_bound == True
    theta = np.array([0, 1])

    # Candidate selection
    pt.propagate_bounds(
        theta=theta,
        tree_dataset_dict={"all":candidate_dataset},
        n_safety=len(safety_features),
        model=model,
        branch="candidate_selection",
        regime="supervised_learning",
        sub_regime="regression"

    )
    assert pt.root.lower == float("-inf")  # not bound_computed
    assert pt.root.upper == pytest.approx(-0.932847)
    pt.reset_base_node_dict(reset_data=True)
    # Safety test
    pt.propagate_bounds(
        theta=theta,
        tree_dataset_dict={"all":safety_dataset},
        model=model,
        branch="safety_test",
        regime="supervised_learning",
        sub_regime="regression"
    )
    assert pt.root.lower == float("-inf")  # not computed
    assert pt.root.upper == pytest.approx(-0.947693)

    # Next, two sided bound
    constraint_str = "abs(Mean_Squared_Error) - 2.0"


    pt = ParseTree(deltas[0], regime="supervised_learning", sub_regime="regression")
    # pt.create_from_ast(constraint_str)
    # pt.assign_bounds_needed()
    # pt.assign_deltas(weight_method="equal")
    # pt.assign_infl_factors()
    pt.build_tree(constraint_str)

    assert pt.n_nodes == 4
    assert pt.n_base_nodes == 1
    assert len(pt.base_node_dict) == 1
    theta = np.array([0, 1])

    # Candidate selection
    pt.propagate_bounds(
        theta=theta,
        tree_dataset_dict={"all":candidate_dataset},
        n_safety=len(safety_features),
        model=model,
        branch="candidate_selection",
        regime="supervised_learning",
        sub_regime="regression"
    )

    # assert pt.root.lower == float('-inf') # not bound_computed
    assert pt.root.upper == pytest.approx(-0.900307)
    pt.reset_base_node_dict(reset_data=True)
    # Safety test
    pt.propagate_bounds(
        theta=theta,
        tree_dataset_dict={"all":safety_dataset},
        model=model,
        branch="safety_test",
        regime="supervised_learning",
        sub_regime="regression"
    )
    # assert pt.root.lower == float('-inf') # not computed
    assert pt.root.upper == pytest.approx(-0.930726)

def test_ttest_bound_listdata(simulated_regression_dataset_aslists):
    # dummy data for linear regression

    # First, single sided bound (MSE only needs upper bound)
    constraint_strs = ["Mean_Squared_Error - 2.0"]
    deltas = [0.05]
    frac_data_in_safety = 0.6

    (
        dataset,
        model,
        primary_objective,
        parse_trees,
    ) = simulated_regression_dataset_aslists(constraint_strs, deltas)

    features = dataset.features
    labels = dataset.labels
    n_points_tot = dataset.num_datapoints
    n_candidate = int(round(n_points_tot * (1.0 - frac_data_in_safety)))
    candidate_features = [x[:n_candidate] for x in features]
    safety_features = [x[n_candidate:] for x in features]

    candidate_labels = labels[:n_candidate]
    safety_labels = labels[n_candidate:]

    candidate_dataset = SupervisedDataSet(
        features=candidate_features,
        labels=candidate_labels,
        sensitive_attrs=[],
        num_datapoints=len(candidate_labels),
        meta=dataset.meta,
    )

    safety_dataset = SupervisedDataSet(
        features=safety_features,
        labels=safety_labels,
        sensitive_attrs=[],
        num_datapoints=len(safety_labels),
        meta=dataset.meta,
    )

    pt = ParseTree(deltas[0], regime="supervised_learning", sub_regime="regression")
    pt.build_tree(constraint_strs[0])

    assert pt.n_nodes == 3
    assert pt.n_base_nodes == 1
    assert len(pt.base_node_dict) == 1
    assert pt.root.name == "sub"
    assert pt.root.left.will_lower_bound == False
    assert pt.root.left.will_upper_bound == True
    theta = np.array([0, 1, 2])

    # Candidate selection
    pt.propagate_bounds(
        theta=theta,
        tree_dataset_dict={"all":candidate_dataset},
        n_safety=len(safety_labels),
        model=model,
        branch="candidate_selection",
        regime="supervised_learning",
        sub_regime="regression"
    )
    assert pt.root.lower == float("-inf")  # not bound_computed
    assert pt.root.upper == pytest.approx(12.341009549)
    pt.reset_base_node_dict(reset_data=True)
    # Safety test
    pt.propagate_bounds(
        theta=theta,
        tree_dataset_dict={"all":safety_dataset},
        model=model,
        branch="safety_test",
        regime="supervised_learning",
        sub_regime="regression"
    )
    assert pt.root.lower == float("-inf")  # not computed
    assert pt.root.upper == pytest.approx(13.1514499)

def test_ttest_bound_infl_factors(simulated_regression_dataset):
    # dummy data for linear regression

    # First, single sided bound (MSE only needs upper bound)
    constraint_strs = ["Mean_Squared_Error - 2.0"]
    deltas = [0.05]
    frac_data_in_safety = 0.6

    (dataset, model, primary_objective, parse_trees) = simulated_regression_dataset(
        constraint_strs, deltas
    )

    features = dataset.features
    labels = dataset.labels

    (
        candidate_features,
        safety_features,
        candidate_labels,
        safety_labels,
    ) = train_test_split(features, labels, test_size=frac_data_in_safety, shuffle=False)

    candidate_dataset = SupervisedDataSet(
        features=candidate_features,
        labels=candidate_labels,
        sensitive_attrs=[],
        num_datapoints=len(candidate_features),
        meta=dataset.meta,
    )

    safety_dataset = SupervisedDataSet(
        features=safety_features,
        labels=safety_labels,
        sensitive_attrs=[],
        num_datapoints=len(safety_features),
        meta=dataset.meta,
    )

    # Build tree with custom bound inflation factor
    pt = ParseTree(deltas[0], regime="supervised_learning", sub_regime="regression")
    pt.build_tree(constraint_strs[0],infl_factors=1)

    assert pt.n_nodes == 3
    assert pt.n_base_nodes == 1
    assert len(pt.base_node_dict) == 1
    assert pt.root.name == "sub"
    assert pt.root.left.will_lower_bound == False
    assert pt.root.left.will_upper_bound == True
    theta = np.array([0, 1])

    # Mock bound propagation in a single iteration of candidate selection
    # The upper bound should be different than the one we get with default inflation factor
    pt.propagate_bounds(
        theta=theta,
        tree_dataset_dict={"all":candidate_dataset},
        n_safety=len(safety_features),
        model=model,
        branch="candidate_selection",
        regime="supervised_learning",
        sub_regime="regression"
    )
    assert pt.root.lower == float("-inf")  # not bound_computed
    assert pt.root.upper == pytest.approx(-1.0175258779025966)
    pt.reset_base_node_dict(reset_data=True)
    # Mock Safety test, the result should be the same because bound inflation factor is not used in safety test
    pt.propagate_bounds(
        theta=theta,
        tree_dataset_dict={"all":safety_dataset},
        model=model,
        branch="safety_test",
        regime="supervised_learning",
        sub_regime="regression"
    )
    assert pt.root.lower == float("-inf")  # not computed
    assert pt.root.upper == pytest.approx(-0.947693)

    # Next, two sided bound
    constraint_str = "abs(Mean_Squared_Error) - 2.0"

    pt = ParseTree(deltas[0], regime="supervised_learning", sub_regime="regression")
    pt.build_tree(constraint_str,infl_factor_method="manual",infl_factors=[2,2])

    assert pt.n_nodes == 4
    assert pt.n_base_nodes == 1
    assert len(pt.base_node_dict) == 1
    theta = np.array([0, 1])

    # Candidate selection
    pt.propagate_bounds(
        theta=theta,
        tree_dataset_dict={"all":candidate_dataset},
        n_safety=len(safety_features),
        model=model,
        branch="candidate_selection",
        regime="supervised_learning",
        sub_regime="regression"
    )

    assert pt.root.upper == pytest.approx(-0.900307)
    pt.reset_base_node_dict(reset_data=True)
    # Safety test
    pt.propagate_bounds(
        theta=theta,
        tree_dataset_dict={"all":safety_dataset},
        model=model,
        branch="safety_test",
        regime="supervised_learning",
        sub_regime="regression"
    )   
    assert pt.root.upper == pytest.approx(-0.930726)

    # Now with custom values we should get different CS results but same ST result
    pt = ParseTree(deltas[0], regime="supervised_learning", sub_regime="regression")
    pt.build_tree(constraint_str,infl_factor_method="manual",infl_factors=[0.5,3])

    assert pt.n_nodes == 4
    assert pt.n_base_nodes == 1
    assert len(pt.base_node_dict) == 1
    theta = np.array([0, 1])

    # Candidate selection
    pt.propagate_bounds(
        theta=theta,
        tree_dataset_dict={"all":candidate_dataset},
        n_safety=len(safety_features),
        model=model,
        branch="candidate_selection",
        regime="supervised_learning",
        sub_regime="regression"
    )

    assert pt.root.upper == pytest.approx(-0.79935738)
    pt.reset_base_node_dict(reset_data=True)
    # Safety test
    pt.propagate_bounds(
        theta=theta,
        tree_dataset_dict={"all":safety_dataset},
        model=model,
        branch="safety_test",
        regime="supervised_learning",
        sub_regime="regression"
    )
    assert pt.root.upper == pytest.approx(-0.930726)

def test_bad_bound_method(simulated_regression_dataset):
    # dummy data for linear regression
    np.random.seed(0)
    numPoints = 1000

    # First, single sided bound (MSE only needs upper bound)
    constraint_strs = ["Mean_Squared_Error - 2.0"]
    deltas = [0.05]
    frac_data_in_safety = 0.6

    (dataset, model, primary_objective, parse_trees) = simulated_regression_dataset(
        constraint_strs, deltas
    )

    features = dataset.features
    labels = dataset.labels

    (
        candidate_features,
        safety_features,
        candidate_labels,
        safety_labels,
    ) = train_test_split(features, labels, test_size=frac_data_in_safety, shuffle=False)

    candidate_dataset = SupervisedDataSet(
        features=candidate_features,
        labels=candidate_labels,
        sensitive_attrs=[],
        num_datapoints=len(candidate_features),
        meta=dataset.meta,
    )

    safety_dataset = SupervisedDataSet(
        features=safety_features,
        labels=safety_labels,
        sensitive_attrs=[],
        num_datapoints=len(safety_features),
        meta=dataset.meta,
    )

    # First, single sided bound (MSE only needs upper bound)

    pt = ParseTree(deltas[0], regime="supervised_learning", sub_regime="regression")
    pt.create_from_ast(constraint_strs[0])
    pt.assign_bounds_needed()
    pt.assign_deltas(weight_method="equal")

    assert pt.n_nodes == 3
    assert pt.n_base_nodes == 1
    assert len(pt.base_node_dict) == 1
    assert pt.root.name == "sub"
    assert pt.root.left.will_lower_bound == False
    assert pt.root.left.will_upper_bound == True
    theta = np.array([0, 1])

    # Candidate selection
    bound_method = "bad-method"
    pt.base_node_dict["Mean_Squared_Error"]["bound_method"] = bound_method
    with pytest.raises(NotImplementedError) as excinfo:
        pt.propagate_bounds(
            theta=theta,
            tree_dataset_dict={"all":candidate_dataset},
            n_safety=len(safety_features),
            model=model,
            branch="candidate_selection",
            regime="supervised_learning",
            sub_regime="regression"
        )

    error_str = f"Bounding method {bound_method} is not supported"
    assert str(excinfo.value) == error_str

    pt.reset_base_node_dict(reset_data=True)
    pt.base_node_dict["Mean_Squared_Error"]["bound_method"] = bound_method
    # Safety test
    with pytest.raises(NotImplementedError) as excinfo:
        pt.propagate_bounds(
            theta=theta,
            tree_dataset_dict={"all":safety_dataset},
            model=model,
            branch="safety_test",
            regime="supervised_learning",
            sub_regime="regression"
        )

    error_str = f"Bounding method {bound_method} is not supported"
    assert str(excinfo.value) == error_str

    # Next, other side single sided bound (MSE only needs lower bound)
    constraint_str = "1.0 - Mean_Squared_Error"
    delta = 0.05

    pt = ParseTree(delta, regime="supervised_learning", sub_regime="regression")
    pt.create_from_ast(constraint_str)
    pt.assign_bounds_needed()
    pt.assign_deltas(weight_method="equal")
    pt.base_node_dict["Mean_Squared_Error"]["bound_method"] = bound_method
    theta = np.array([0, 1])

    # Candidate selection
    with pytest.raises(NotImplementedError) as excinfo:
        pt.propagate_bounds(
            theta=theta,
            tree_dataset_dict={"all":candidate_dataset},
            n_safety=len(safety_features),
            model=model,
            branch="candidate_selection",
            regime="supervised_learning",
            sub_regime="regression"
        )

    error_str = f"Bounding method {bound_method} is not supported"
    assert str(excinfo.value) == error_str

    pt.reset_base_node_dict(reset_data=True)
    pt.base_node_dict["Mean_Squared_Error"]["bound_method"] = bound_method
    # Safety test
    with pytest.raises(NotImplementedError) as excinfo:
        pt.propagate_bounds(
            theta=theta,
            tree_dataset_dict={"all":safety_dataset},
            model=model,
            branch="safety_test",
            regime="supervised_learning",
            sub_regime="regression"
        )

    error_str = f"Bounding method {bound_method} is not supported"
    assert str(excinfo.value) == error_str

    # Now, two sided bound on leaf node
    constraint_str = "abs(Mean_Squared_Error) - 2.0"
    delta = 0.05

    pt = ParseTree(delta, regime="supervised_learning", sub_regime="regression")
    pt.create_from_ast(constraint_str)
    pt.assign_bounds_needed()
    pt.assign_deltas(weight_method="equal")
    pt.base_node_dict["Mean_Squared_Error"]["bound_method"] = bound_method

    # Candidate selection
    bound_method = "bad-method"
    with pytest.raises(NotImplementedError) as excinfo:
        pt.propagate_bounds(
            theta=theta,
            tree_dataset_dict={"all":candidate_dataset},
            n_safety=len(safety_features),
            model=model,
            branch="candidate_selection",
            regime="supervised_learning",
            sub_regime="regression"
        )

    error_str = f"Bounding method {bound_method} is not supported"
    assert str(excinfo.value) == error_str

    pt.reset_base_node_dict(reset_data=True)
    pt.base_node_dict["Mean_Squared_Error"]["bound_method"] = bound_method
    # Safety test
    with pytest.raises(NotImplementedError) as excinfo:
        pt.propagate_bounds(
            theta=theta,
            tree_dataset_dict={"all":safety_dataset},
            model=model,
            branch="safety_test",
            regime="supervised_learning",
            sub_regime="regression"
        )

    error_str = f"Bounding method {bound_method} is not supported"
    assert str(excinfo.value) == error_str

def test_evaluate_constraint(
    simulated_regression_dataset, gpa_classification_dataset, RL_gridworld_dataset,
    custom_text_spec
):
    # Evaluate constraint mean, not the bound
    # test all of the statistics in all regimes
    import copy

    ### Regression
    constraint_strs = ["Mean_Squared_Error - 2.0"]
    deltas = [0.05]
    frac_data_in_safety = 0.6

    (dataset, model, primary_objective, parse_trees) = simulated_regression_dataset(
        constraint_strs, deltas
    )

    features = dataset.features
    labels = dataset.labels

    (
        candidate_features,
        safety_features,
        candidate_labels,
        safety_labels,
    ) = train_test_split(features, labels, test_size=frac_data_in_safety, shuffle=False)

    # MSE
    pt = ParseTree(deltas[0], regime="supervised_learning", sub_regime="regression")
    pt.create_from_ast(constraint_strs[0])

    pt.assign_bounds_needed()
    pt.assign_deltas(weight_method="equal")
    assert pt.n_nodes == 3
    assert pt.n_base_nodes == 1
    assert len(pt.base_node_dict) == 1

    theta = np.array([0, 1])
    pt.evaluate_constraint(
        theta=theta,
        tree_dataset_dict={"all":dataset},
        model=model,
        regime="supervised_learning",
        branch="safety_test",
        sub_regime="regression"
    )

    assert pt.root.value == pytest.approx(-1.06248)

    ### Classification
    constraint_str = (
        "(abs(PR) + exp(NR*2) + FPR/4.0 + max(FNR,TPR) + min(TNR,0.5)) - 10.0"
    )
    constraint_strs = [constraint_str]
    deltas = [0.05]

    (dataset, model, primary_objective, parse_trees) = gpa_classification_dataset(
        constraint_strs=constraint_strs, deltas=deltas
    )

    theta = np.zeros(10)
    pt = parse_trees[0]
    pt.evaluate_constraint(
        theta=theta,
        tree_dataset_dict={"all":dataset},
        model=model,
        regime="supervised_learning",
        branch="safety_test",
        sub_regime="classification"
    )
    assert pt.root.value == pytest.approx(-5.656718)

    ### RL
    constraint_str = "J_pi_new_IS >= -0.25"
    constraint_strs = [constraint_str]
    deltas = [0.05]
    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,
        deltas,
        regime="reinforcement_learning",
        sub_regime="all",
        columns=[],
        delta_weight_method="equal",
    )

    (dataset, policy, env_kwargs, primary_objective) = RL_gridworld_dataset()

    frac_data_in_safety = 0.6

    model = RL_model(policy=policy, env_kwargs=env_kwargs)
    theta_init = model.policy.get_params()
    pt = parse_trees[0]
    # The default is off-policy evaluation, which should give us one answer
    pt.evaluate_constraint(
        theta=theta_init,
        tree_dataset_dict={"all":dataset},
        model=model,
        regime="reinforcement_learning",
        branch="safety_test",
        sub_regime="all"
    )
    assert pt.root.right.value == pytest.approx(-0.3411361059961765)
    # Now do on-policy evaluation (used as default in experiments)
    # Need new episodes for this. Just create altered versions of the behavior
    # episodes
    new_episodes = copy.deepcopy(dataset.episodes)
    for i in range(100):
        if new_episodes[i].rewards[-1] == 1:
            new_episodes[i].rewards[-1] = 100

    new_dataset = RLDataSet(episodes=new_episodes, meta=dataset.meta)
    # Make new parse trees
    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,
        deltas,
        regime="reinforcement_learning",
        sub_regime="all",
        columns=[],
        delta_weight_method="equal",
    )
    pt = parse_trees[0]
    pt.evaluate_constraint(
        theta=theta_init,
        tree_dataset_dict={"all":new_dataset},
        model=model,
        regime="reinforcement_learning",
        branch="safety_test",
        on_policy=True,
        sub_regime="all"
    )
    assert pt.root.right.value == pytest.approx(12.983593429599098)

    np.random.seed(0)

    ### Custom regime
    custom_spec = custom_text_spec()
    pt = custom_spec.parse_trees[0]
    theta_init = np.array([-1.0,0.0,1.0])
    pt.evaluate_constraint(
        theta=theta_init,
        tree_dataset_dict={"all":custom_spec.dataset},
        model=custom_spec.model,
        regime="custom",
        branch="safety_test",
        sub_regime=None
    )
    assert pt.root.left.value == 10.0


def test_reset_parse_tree():
    constraint_str = "(FPR + FNR) - 0.5"
    delta = 0.05

    pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")
    pt.create_from_ast(constraint_str)
    pt.assign_deltas(weight_method="equal")
    assert pt.n_base_nodes == 2
    assert len(pt.base_node_dict) == 2
    assert pt.base_node_dict["FPR"]["bound_computed"] == False
    assert pt.base_node_dict["FPR"]["lower"] == float("-inf")
    assert pt.base_node_dict["FPR"]["upper"] == float("inf")
    assert pt.base_node_dict["FPR"]["bound_method"] == "ttest"  # the default
    assert pt.base_node_dict["FNR"]["lower"] == float("-inf")
    assert pt.base_node_dict["FNR"]["upper"] == float("inf")
    assert pt.base_node_dict["FNR"]["bound_computed"] == False
    assert pt.base_node_dict["FNR"]["bound_method"] == "ttest"  # the default

    pt.base_node_dict["FPR"]["bound_method"] = "random"
    pt.base_node_dict["FNR"]["bound_method"] = "random"

    # propagate bounds
    pt.propagate_bounds()
    assert len(pt.base_node_dict) == 2
    assert pt.base_node_dict["FPR"]["bound_computed"] == True
    assert pt.base_node_dict["FNR"]["bound_computed"] == True
    assert pt.base_node_dict["FPR"]["lower"] >= 0
    assert pt.base_node_dict["FPR"]["upper"] > 0
    assert pt.base_node_dict["FNR"]["lower"] >= 0
    assert pt.base_node_dict["FNR"]["upper"] > 0

    # # reset the node dict
    pt.reset_base_node_dict()
    assert len(pt.base_node_dict) == 2
    assert pt.base_node_dict["FPR"]["bound_computed"] == False
    assert pt.base_node_dict["FNR"]["bound_computed"] == False
    assert pt.base_node_dict["FPR"]["lower"] == float("-inf")
    assert pt.base_node_dict["FPR"]["upper"] == float("inf")
    assert pt.base_node_dict["FNR"]["lower"] == float("-inf")
    assert pt.base_node_dict["FNR"]["upper"] == float("inf")

def test_single_conditional_columns_propagated(
    gpa_regression_dataset,
):
    np.random.seed(0)
    # Supervised learning
    constraint_strs = ["abs(Mean_Error|[M]) - 0.1"]
    deltas = [0.05]
    (dataset, model, primary_objective, parse_trees) = gpa_regression_dataset(
        constraint_strs, deltas
    )

    pt = ParseTree(
        deltas[0],
        regime="supervised_learning",
        sub_regime="regression",
        columns=dataset.meta.sensitive_col_names,
    )

    pt.create_from_ast(constraint_strs[0])
    pt.assign_deltas(weight_method="equal")

    # propagate the bounds with example theta value
    # theta = np.hstack([np.array([0.0,0.0]),np.random.uniform(-0.05,0.05,10)])
    theta = np.random.uniform(-0.05, 0.05, 10)
    pt.propagate_bounds(
        theta=theta,
        tree_dataset_dict={"all":dataset},
        model=model,
        branch="safety_test",
        regime="supervised_learning",
        sub_regime="regression"
    )
    assert pt.root.lower == pytest.approx(61.9001779655)
    assert pt.root.upper == pytest.approx(62.1362236720)

    assert len(pt.base_node_dict["Mean_Error | [M]"]["data_dict"]["features"]) == 22335

    # Reinforcement learning
    from seldonian.RL.RL_model import RL_model
    from seldonian.RL.Agents.Policies.Softmax import DiscreteSoftmax
    from seldonian.RL.Env_Description import Spaces, Env_Description

    data_pth = "static/datasets/RL/gridworld/gridworld_100episodes.pkl"

    episodes = load_pickle(data_pth)
    RL_meta = RLMetaData(
        all_col_names=["episode_index", "O", "A", "R", "pi_b"],
        sensitive_col_names=["M", "F"],
    )
    M = np.random.randint(0, 2, len(episodes))
    F = 1 - M
    sensitive_attrs = np.hstack((M.reshape(-1, 1), F.reshape(-1, 1)))
    RL_dataset = RLDataSet(
        episodes=episodes, sensitive_attrs=sensitive_attrs, meta=RL_meta
    )

    # Initialize policy
    num_states = 9
    observation_space = Spaces.Discrete_Space(0, num_states - 1)
    action_space = Spaces.Discrete_Space(0, 3)
    env_description = Env_Description.Env_Description(observation_space, action_space)
    policy = DiscreteSoftmax(
        hyperparam_and_setting_dict={}, env_description=env_description
    )
    env_kwargs = {"gamma": 0.9}
    RLmodel = RL_model(policy=policy, env_kwargs=env_kwargs)

    RL_constraint_strs = ["(J_pi_new_IS | [M]) >= -0.25"]
    RL_deltas = [0.05]

    RL_pt = ParseTree(
        RL_deltas[0],
        regime="reinforcement_learning",
        sub_regime="all",
        columns=RL_dataset.meta.sensitive_col_names,
    )

    RL_pt.create_from_ast(RL_constraint_strs[0])
    RL_pt.assign_deltas(weight_method="equal")

    # propagate the bounds with example theta value
    # theta = np.hstack([np.array([0.0,0.0]),np.random.uniform(-0.05,0.05,10)])
    RL_theta = np.random.uniform(-0.05, 0.05, (9, 4))
    RL_pt.propagate_bounds(
        theta=RL_theta,
        tree_dataset_dict={"all":RL_dataset},
        model=RLmodel,
        branch="safety_test",
        regime="reinforcement_learning",
        sub_regime="all"
    )
    assert RL_pt.root.lower == pytest.approx(-0.00556309)
    assert RL_pt.root.upper == pytest.approx(0.333239520)

    assert len(RL_pt.base_node_dict["J_pi_new_IS | [M]"]["data_dict"]["episodes"]) == 52

def test_build_tree():
    """Test the convenience function that builds the tree,
    weights deltas, and assigns bounds all in one"""

    constraint_str = "(FPR + FNR) - 0.5"
    delta = 0.05

    pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")

    # build the tree the original way
    pt.create_from_ast(constraint_str)
    pt.assign_deltas(weight_method="equal")
    assert pt.n_base_nodes == 2
    assert len(pt.base_node_dict) == 2
    assert pt.base_node_dict["FPR"]["bound_computed"] == False
    assert pt.base_node_dict["FPR"]["lower"] == float("-inf")
    assert pt.base_node_dict["FPR"]["upper"] == float("inf")
    assert pt.base_node_dict["FNR"]["lower"] == float("-inf")
    assert pt.base_node_dict["FNR"]["upper"] == float("inf")
    assert pt.base_node_dict["FNR"]["bound_computed"] == False

    ##### build the tree with the convenience function
    pt2 = ParseTree(delta, regime="supervised_learning", sub_regime="classification")
    pt2.build_tree(constraint_str=constraint_str, delta_weight_method="equal")
    assert pt2.n_base_nodes == 2
    assert len(pt2.base_node_dict) == 2
    assert pt2.base_node_dict["FPR"]["bound_computed"] == False
    assert pt2.base_node_dict["FPR"]["lower"] == float("-inf")
    assert pt2.base_node_dict["FPR"]["upper"] == float("inf")
    assert pt2.base_node_dict["FNR"]["lower"] == float("-inf")
    assert pt2.base_node_dict["FNR"]["upper"] == float("inf")
    assert pt2.base_node_dict["FNR"]["bound_computed"] == False

def test_bad_delta():
    """Test that supplying delta not in (0,1) raises a ValueError"""
    constraint_str = "FPR <= 0.1"
    delta = 0.0
    with pytest.raises(ValueError) as excinfo:
        pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")

    error_str = "delta must be in (0,1)"
    assert str(excinfo.value) == error_str

    delta = 1.0
    with pytest.raises(ValueError) as excinfo:
        pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")

    error_str = "delta must be in (0,1)"
    assert str(excinfo.value) == error_str

    delta = -2.5
    with pytest.raises(ValueError) as excinfo:
        pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")

    error_str = "delta must be in (0,1)"
    assert str(excinfo.value) == error_str

    delta = 7
    with pytest.raises(ValueError) as excinfo:
        pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")

    error_str = "delta must be in (0,1)"
    assert str(excinfo.value) == error_str

def test_e_assigned_as_constant_node():
    constraint_str = "FPR <= e*0.05"
    delta = 0.05

    pt = ParseTree(delta, regime="supervised_learning", sub_regime="classification")
    pt.create_from_ast(constraint_str)
    assert pt.root.right.left.name == "e"
    assert isinstance(pt.root.right.left, ConstantNode)
