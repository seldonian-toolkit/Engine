import copy
import autograd.numpy as np  # Thinly-wrapped version of Numpy
from autograd import grad, jacobian, elementwise_grad as egrad

import warnings
from seldonian.warnings.custom_warnings import *


def setup_gradients(gradient_library, primary_objective, upper_bounds_function):
    """Wrapper to obtain the gradient functions
    of the primary objective and upper bounds function
    given a gradient library

    :param gradient_library: The name of the library to use for computing
        automatic gradients.
    :type gradient_library: str, defaults to "autograd"
    :param primary_objective: Primary objective function
    :param upper_bounds_function: Function for computing upper bounds
        on the constraints
    """
    if gradient_library == "autograd":
        grad_primary_theta = grad(primary_objective, argnum=0)
        grad_upper_bound_theta = jacobian(upper_bounds_function, argnum=0)
    else:
        raise NotImplementedError(
            f"gradient library: {gradient_library}" " not supported"
        )
    return grad_primary_theta, grad_upper_bound_theta


def gradient_descent_adam(
    primary_objective,
    n_constraints,
    upper_bounds_function,
    theta_init,
    lambda_init,
    batch_calculator,
    n_batches,
    batch_size=100,
    n_epochs=1,
    alpha_theta=0.05,
    alpha_lamb=0.05,
    beta_velocity=0.9,
    beta_rmsprop=0.9,
    gradient_library="autograd",
    clip_theta=None,
    verbose=False,
    debug=False,
    **kwargs,
):
    """Implements KKT optimization, i.e. simultaneous gradient descent/ascent using
    the Adam optimizer on a Lagrangian:
    L(theta,lambda) = f(theta) + lambda*g(theta),
    where f is the primary objective, lambda is a vector of
    Lagrange multipliers, and g is a vector of the
    upper bound functions. Gradient descent is done on theta
    and gradient ascent is done on lambda to find the saddle
    points of L. We only are interested in the optimal theta.
    Being part of candidate selection,
    If a nan or inf occurs during the optimization, NSF is returned. 
    The optimal solution is defined as the feasible solution (i.e.
    all constraints satisfied), that has the smallest primary objective value. 

    :param primary_objective: The objective function that would
        be solely optimized in the absence of behavioral constraints,
        i.e., the loss function
    :type primary_objective: function or class method
    :param n_constraints: The number of constraints
    :param upper_bounds_function: The function that calculates
        the upper bounds on the constraints
    :type upper_bounds_function: function or class method
    :param theta_init: Initial model weights
    :type theta_init: numpy ndarray
    :param lambda_init: Initial values for Lagrange multiplier terms
    :type theta_init: float
    :param batch_calculator: A function/class method that sets the current batch
        and returns whether the batch is viable for generating 
        a candidate solution
    :param batches: The number of batches per epoch
    :type batches: int
    :param n_epochs: The number of epochs to run
    :type n_epochs: int
    :param alpha_theta: Initial learning rate for theta
    :type alpha_theta: float
    :param alpha_lamb: Initial learning rate for lambda
    :type alpha_lamb: float
    :param beta_velocity: Exponential decay rate for velocity term
    :type beta_velocity: float
    :param beta_rmsprop: Exponential decay rate for rmsprop term
    :type beta_rmsprop: float
    :param num_iters: The number of iterations of gradient descent to run
    :type num_iters: int
    :param gradient_library: The name of the library to use for computing
        automatic gradients.
    :type gradient_library: str, defaults to "autograd"
    :param clip_theta: Optional, the min and max values 
        between which to clip all values in the theta vector
    :type clip_theta: tuple, list or numpy.ndarray, defaults to None
    :param verbose: Boolean flag to control verbosity
    :param debug: Boolean flag to print out info useful for debugging

    :return: solution, a dictionary containing the candidate solution and values of 
        the parameters of the KKT optimization at each step.
    :rtype: dict
    """

    # initialize theta, lambda
    theta = theta_init
    # If lambda provided as a float, make a vector
    # with this value for all constraints
    if type(lambda_init) == float:
        lamb = np.repeat(lambda_init, n_constraints)
    else:
        lamb = np.array(lambda_init)

    if len(lamb) != n_constraints:
        raise RuntimeError(
            "lambda has wrong shape. Shape must be (n_constraints,), "
            f"but shape is {lamb.shape}"
        )

    # initialize Adam parameters
    velocity_theta, velocity_lamb = 0.0, 0.0
    s_theta, s_lamb = 0.0, 0.0
    rms_offset = 1e-6  # small offset to make sure we don't take 1/sqrt(very small) in weight update

    # Initialize params for tracking best solution
    best_primary = np.inf  # minimizing f so want it to be lowest possible
    best_index = 0
    candidate_solution = None

    # If we never enter feasible set, we still need to know what the solution was
    # when g was minimum
    found_feasible_solution = False
    # Store values at each step in gradient descent, if requested
    theta_vals = []
    lamb_vals = []
    L_vals = []
    f_vals = []  # primary
    g_vals = []  # constraint upper bound values
    # min(sqrt(g**2)) used to select candidate solution if no feasible solution found
    best_g_norm = np.inf
    best_index_g_norm = 0

    # Get df/dtheta and dg/dtheta automatic gradients
    (grad_primary_theta, grad_upper_bound_theta) = setup_gradients(
        gradient_library, primary_objective, upper_bounds_function
    )

    # It is possible that the user provided the function df/dtheta,
    # which can often speed up computing the gradients.
    # In that case, override the automatic gradient function
    if "primary_gradient" in kwargs:
        grad_primary_theta = kwargs["primary_gradient"]

    # Start gradient descent
    gd_index = 0
    if verbose:
        n_iters_tot = n_epochs * n_batches
        print(
            f"Have {n_epochs} epochs and {n_batches} batches of size {batch_size} "
            f"for a total of {n_iters_tot} iterations"
        )

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            if verbose:
                if batch_index % 10 == 0:
                    print(f"Epoch: {epoch}, batch iteration {batch_index}")
            is_small_batch = batch_calculator(batch_index, batch_size, epoch, n_batches)
            primary_val = primary_objective(theta)
            g_vec = upper_bounds_function(theta)
            # Check if the 2-norm is smallest so far.
            # We will use the smallest overall as a backup
            # candidate solution in case we don't find a feasible solution
            g_norm = np.linalg.norm(g_vec)
            if g_norm < best_g_norm:
                best_g_norm = g_norm
                best_index_g_norm = gd_index
                candidate_solution_best_g_norm = np.copy(theta)

            L_val = primary_val + sum(lamb * g_vec)

            if debug:
                print(
                    "epoch,batch_i,overall_i,f,g,theta,lambda:",
                    epoch,
                    batch_index,
                    gd_index,
                    primary_val,
                    g_vec,
                    theta,
                    lamb,
                )
                print()

            # Check if this is best feasible value so far
            if (
                (not is_small_batch)
                and all([g <= 0 for g in g_vec])
                and primary_val < best_primary
            ):
                found_feasible_solution = True
                best_index = gd_index
                best_primary = primary_val
                best_lamb = lamb
                best_g_vec = g_vec
                best_L = L_val
                candidate_solution = np.copy(theta)

            # store values
            # theta_vals.append(np.copy(theta))
            lamb_vals.append(np.copy(lamb))
            f_vals.append(primary_val)
            g_vals.append(g_vec)
            L_vals.append(L_val)

            # if nans or infs appear in any quantities,
            # then stop gradient descent and return NSF
            if (
                np.isinf(primary_val)
                or np.isnan(primary_val)
                or np.isinf(lamb).any()
                or np.isnan(lamb).any()
                or np.isinf(theta).any()
                or np.isnan(theta).any()
                or np.isinf(g_vec).any()
                or np.isnan(g_vec).any()
            ):
                warning_msg = (
                    "Warning: a nan or inf was found during "
                    "gradient descent. Stopping prematurely "
                    "and returning NSF."
                )
                warnings.warn(warning_msg)
                candidate_solution = "NSF"
                break

            # Obtain gradients of both terms in Lagrangian
            # at current values of theta and lambda
            grad_primary_theta_val = grad_primary_theta(theta)
            gu_theta_vec = grad_upper_bound_theta(theta)

            grad_secondary_theta_val_vec = (
                gu_theta_vec * lamb[:, None]
            )  ## to multiply each row of gu_theta_vec by elements of lamb
            gradient_theta = grad_primary_theta_val + np.sum(
                grad_secondary_theta_val_vec, axis=0
            )

            # gradient w.r.t. to lambda is just g
            gradient_lamb_vec = g_vec

            # Momementum term
            velocity_theta = (
                beta_velocity * velocity_theta + (1.0 - beta_velocity) * gradient_theta
            )

            # RMS prop term
            s_theta = beta_rmsprop * s_theta + (1.0 - beta_rmsprop) * pow(
                gradient_theta, 2
            )

            # bias-correction
            velocity_theta /= 1 - pow(beta_velocity, gd_index + 1)
            s_theta /= 1 - pow(beta_rmsprop, gd_index + 1)

            # update weights
            theta -= (
                alpha_theta * velocity_theta / (np.sqrt(s_theta) + rms_offset)
            )  # gradient descent
            lamb += alpha_lamb * gradient_lamb_vec  # element wise update

            # Clip theta if specified
            if clip_theta:
                th_min, th_max = clip_theta
                theta = np.clip(theta, th_min, th_max)
            # If any values in lambda vector dip below 0, force them to be zero
            lamb[lamb < 0] = 0

            gd_index += 1
        else:  # only executed if inner loop did not break
            continue
        break  # only executed if inner loop broke

    solution = {}
    solution_found = True

    # If theta never entered feasible set pick best g
    if not found_feasible_solution:
        if candidate_solution == "NSF":
            if debug:
                print("NaN or Inf appeared in gradient descent terms " "Returning NSF")
            best_index = None
            best_primary = None
            best_lamb = None
            best_g_vec = None
            best_L = None
        else:
            if debug:
                print(
                    "Never found feasible solution. "
                    "Returning solution with lowest sqrt(|g|**2)"
                )
            # best g is when norm of g is minimized
            best_primary = f_vals[best_index_g_norm]
            best_lamb = lamb_vals[best_index_g_norm]
            best_g_vec = g_vals[best_index_g_norm]
            best_L = L_vals[best_index_g_norm]
            candidate_solution = candidate_solution_best_g_norm

    solution["candidate_solution"] = candidate_solution
    solution["best_index"] = best_index
    solution["best_f"] = best_primary
    solution["best_g"] = best_g_vec
    solution["best_lamb"] = best_lamb
    solution["best_L"] = best_L
    solution["found_feasible_solution"] = found_feasible_solution
    # solution["theta_vals"] = np.array(theta_vals) # takes up too much disk
    solution["f_vals"] = np.array(f_vals)
    solution["lamb_vals"] = np.array(lamb_vals)
    solution["g_vals"] = np.array(g_vals)
    solution["L_vals"] = np.array(L_vals)

    return solution
