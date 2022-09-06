import copy
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad, jacobian

def setup_gradients(
    gradient_library,
    primary_objective,
    upper_bounds_function):

    if gradient_library == "autograd":
        grad_primary_theta = grad(primary_objective,argnum=0)
        grad_upper_bound_theta = jacobian(upper_bounds_function,argnum=0)
    else:
        raise NotImplementedError(
            f"gradient library: {gradient_library}"
            " not supported")

    return grad_primary_theta,grad_upper_bound_theta

def gradient_descent_adam(
    primary_objective,
    n_constraints,
    upper_bounds_function,
    theta_init,
    lambda_init,
    alpha_theta=0.05,
    alpha_lamb=0.05,
    beta_velocity=0.9,
    beta_rmsprop=0.9,
    num_iters=200,
    gradient_library="autograd",
    store_values=False,
    verbose=False,
    debug=False,
    **kwargs):
    """ Implements simultaneous gradient descent/ascent using 
    the "adam" optimizer on a Lagrangian:
    L(theta,lambda) = f(theta) + lambda*g(theta),
    where f is the primary objective, lambda is a vector of 
    Lagrange multipliers, and g is a vector of the 
    upper bound functions. Gradient descent is done for theta 
    and gradient ascent is done for lambda to find the saddle 
    points of L.

    :param primary_objective: The objective function that would
        be solely optimized in the absence of behavioral constraints,
        i.e. the loss function
    :type primary_objective: function or class method

    :param n_constraints: The number of constraints 

    :param upper_bounds_function: The function that calculates
        the upper bounds on the constraints
    :type upper_bounds_function: function or class method

    :param theta_init: Initial model weights 
    :type theta_init: numpy ndarray

    :param lambda_init: Initial values for Lagrange multiplier terms 
    :type theta_init: float

    :param alpha_theta: Initial learning rate for theta
    :type alpha_theta: float

    :param alpha_lamb: Learning rate for lambda
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

    :param store_values: Whether to include evaluations of various 
        quantities in the solution dictionary
    :type store_values: bool

    :return: solution, a dictionary containing the solution and metadata 
        about the gradient descent run, if store_values==True
    :rtype: dict
    """
    
    # initialize modeling parameters
    theta = theta_init
    lamb = lambda_init.reshape(lambda_init.shape[0],1) # like [[0.5],[0.5],[0.5],...[0.5]]
    if lamb.shape[0] == 1 and lamb.shape[0] != n_constraints:
        # repeat value for each constraint
        lamb = lamb[0][0]*np.ones((n_constraints,1))

    
    velocity_theta, velocity_lamb = 0.0,0.0
    s_theta, s_lamb = 0.0,0.0
    rms_offset = 1e-6 # small offset to make sure we don't take 1/sqrt(very small) in weight update

    best_feasible_primary = np.inf # minimizing f so want it to be lowest possible
    best_index = 0  

    # Get df/dtheta and dg/dtheta
    (grad_primary_theta,
        grad_upper_bound_theta) = setup_gradients(
        gradient_library,
        primary_objective,
        upper_bounds_function)
        
    # It is possible user provided the function df/dtheta,
    # which can often speed up computing the gradients.

    if 'primary_gradient' in kwargs:
        grad_primary_theta = kwargs['primary_gradient']
    
    if store_values: 
        # Store results in lists
        theta_vals = []
        lamb_vals = []
        L_vals = []
        f_vals = [] # primary
        g_vals = [] # constraint upper bound values

    for i in range(num_iters):
        if verbose:
            if i % 10 == 0:
                print(f"Iteration {i}")
        primary_val = primary_objective(theta)
        g_vec = upper_bounds_function(theta)
        g_vec = g_vec.reshape(g_vec.shape[0],1)
        if debug:
            print("it,f,g,theta,lambda:",i,primary_val,g_vec,theta,lamb)
        # Check if this is best feasible value so far
        if all([g<= 0 for g in g_vec]) and primary_val < best_feasible_primary:
            best_feasible_primary = np.copy(primary_val)
            best_feasible_g_vec = np.copy(g_vec)
            best_index = np.copy(i)
            candidate_solution = np.copy(theta)
            # print()

        if store_values:
            theta_vals.append(theta.tolist())
            lamb_vals.append(np.copy(lamb))
            f_vals.append(np.copy(primary_val))
            g_vals.append(np.copy(g_vec))
            
            L_val = primary_val + sum(lamb*g_vec) 
            L_vals.append(L_val)

        # Obtain gradients of both terms in Lagrangian 
        # at current values of theta and lambda
        grad_primary_theta_val = grad_primary_theta(theta)
        gu_theta_vec = grad_upper_bound_theta(theta)
        
        grad_secondary_theta_val_vec = lamb*gu_theta_vec # elementwise mult
        
        # Gradient of sum is sum of gradients
        gradient_theta = grad_primary_theta_val + sum(grad_secondary_theta_val_vec)
        
        # gradient wr.t. to lambda is just g
        gradient_lamb_vec = g_vec

        # Momementum term
        velocity_theta = beta_velocity*velocity_theta + (1.0-beta_velocity)*gradient_theta

        # RMS prop term
        s_theta = beta_rmsprop*s_theta + (1.0-beta_rmsprop)*pow(gradient_theta,2)

        # bias-correction
        velocity_theta /= (1-pow(beta_velocity,i+1))
        s_theta /= (1-pow(beta_rmsprop,i+1))

        # update weights
        theta -= alpha_theta*velocity_theta/(np.sqrt(s_theta)+rms_offset) # gradient descent
        lamb += alpha_lamb*gradient_lamb_vec # element wise update
        
        # If any values in lambda vector dip below 0, force them to be zero
        lamb[lamb<0]=0
        if np.isinf(primary_val) or np.isnan(primary_val) or np.isinf(lamb).any() or np.isnan(lamb).any() or np.isinf(theta).any() or np.isnan(theta).any() or np.isinf(g_vec).any() or np.isnan(g_vec).any():
            break

    solution = {}
    solution_found = True
    if np.isinf(best_feasible_primary):
        solution_found = False
    else:    
        # solution['candidate_solution'] = candidate_solution.tolist()
        solution['candidate_solution'] = candidate_solution
        solution['best_index'] = best_index
        solution['best_feasible_g'] = best_feasible_g_vec
        solution['best_feasible_f'] = best_feasible_primary

    solution['solution_found'] = solution_found
    
    if store_values:
        solution['theta_vals'] = theta_vals
        solution['f_vals'] = f_vals
        solution['g_vals'] = g_vals
        solution['lamb_vals'] = lamb_vals
        solution['L_vals'] = L_vals

    return solution