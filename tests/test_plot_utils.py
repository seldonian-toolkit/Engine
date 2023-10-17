from seldonian.utils.io_utils import load_pickle
from seldonian.utils.plot_utils import plot_gradient_descent

### Begin tests


def test_plot_gradient_descent():
    """Test the function used to plot the
    gradient descent logs for one and multiple
    constraints"""

    path_two_constraints = (
        "static/gradient_descent_logs/candidate_selection_two_constraints.p"
    )
    sol_two_constraints = load_pickle(path_two_constraints)
    fig = plot_gradient_descent(
        solution=sol_two_constraints,
        primary_objective_name="MSE",
        save=False,
        show=False,
    )
    assert len(fig.axes) == 8

    path_single_constraint = (
        "static/gradient_descent_logs/candidate_selection_single_constraint.p"
    )
    sol_single_constraint = load_pickle(path_single_constraint)
    fig = plot_gradient_descent(
        solution=sol_single_constraint,
        primary_objective_name="logistic loss",
        save=False,
        show=False,
    )
    assert len(fig.axes) == 4

    path_single_constraint_NSF = (
        "static/gradient_descent_logs/candidate_selection_single_constraint_NSF.p"
    )
    sol_single_constraint_NSF = load_pickle(path_single_constraint_NSF)
    fig = plot_gradient_descent(
        solution=sol_single_constraint_NSF,
        primary_objective_name="logistic loss",
        save=False,
        show=False,
    )
    assert len(fig.axes) == 4
