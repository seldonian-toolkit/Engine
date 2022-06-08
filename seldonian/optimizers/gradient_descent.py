import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad

def gradient_descent_adam(
    primary_objective,
    upper_bound_function,
    theta_init=np.array([-2.0]),
    lambda_init=0.5,
    alpha_theta=0.05,
    alpha_lamb=0.05,
    beta_velocity=0.9,
    beta_rmsprop=0.9,
    num_iters=200,
    store_values=False,
    verbose=False,
    **kwargs):
    """ Implements gradient descent with "adam" optimizer

    :param primary_objective: The objective function that would
        be solely optimized in the absence of behavioral constraints,
        i.e. the loss function
    :type primary_objective: function or class method

    :param upper_bound_function: The function that calculates
        the upper bound on the constraint
    :type upper_bound_function: function or class method

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

    :param store_values: Whether to include evaluations of various 
        quantities in the solution dictionary
    :type store_values: bool

    :return: solution, a dictionary containing the solution and metadata 
        about the gradient descent run, if store_values==True
    :rtype: dict
    """
    
    # initialize modeling parameters
    theta = theta_init
    lamb = lambda_init
    velocity_theta, velocity_lamb = 0.0,0.0
    s_theta, s_lamb = 0.0,0.0
    rms_offset = 1e-6 # small offset to make sure we don't take 1/sqrt(very small) in weight update

    best_feasible_primary = np.inf # minimizing f so want it to be lowest possible
    best_index = 0  

    # Define a function that returns gradients of LM loss function using Autograd
    # It is possible the user provided a function for the gradient of the primary
    if 'primary_gradient' in kwargs:
        grad_primary_theta = kwargs['primary_gradient']
    else:
        grad_primary_theta = grad(primary_objective,argnum=0)

    grad_upper_bound_theta = grad(upper_bound_function,argnum=0)
    
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
        g_val = upper_bound_function(theta)
        # if 'parse_trees' in kwargs:
        #     pt = kwargs['parse_trees'][0]
        #     graph = pt.make_viz(pt.constraint_str)
        #     graph.view()
        #     input("next")
        
        # Check if this is best feasible value so far
        if g_val <= 0 and primary_val < best_feasible_primary:
            best_feasible_primary = primary_val
            best_feasible_g = g_val
            best_index = i
            candidate_solution = theta

        if store_values:
            theta_vals.append(theta.tolist())
            lamb_vals.append(lamb)
            f_vals.append(primary_val)
            g_vals.append(g_val)
            
            L_val = primary_val + lamb*g_val 
            L_vals.append(L_val)

        # Obtain gradients at current values of theta and lambda
        # Gradient of sum is sum of gradients
        # print(f"theta: {theta}")
        grad_primary_theta_val = grad_primary_theta(theta)
        # print(primary_val,g_val)
        # print("calculating d upper_bound_function / dtheta")
        # print("")
        gu_theta = grad_upper_bound_theta(theta)
        # print(f"d primary/d theta: {grad_primary_theta_val}")
        # print(f"d upper_bound/d theta: {gu_theta}")
        # # print("Done")
        # input("Next")
        # print(f"theta = {theta}")
        # print(f"lambda = {lamb}")
        # print
        # print(f"primary = {primary_val}, g_val = {g_val}")
        # print(f"g upper bound = {g_val}")
        # print(f"d primary/d theta: {grad_primary_theta_val}")
        # print(f"d upper bound/d theta: {gu_theta}")
        # input("next")
        grad_secondary_theta_val = lamb*gu_theta
        gradient_theta = grad_primary_theta_val + grad_secondary_theta_val
        gradient_lamb = g_val

        # Momementum term
        velocity_theta = beta_velocity*velocity_theta + (1.0-beta_velocity)*gradient_theta

        # RMS prop term
        s_theta = beta_rmsprop*s_theta + (1.0-beta_rmsprop)*pow(gradient_theta,2)

        # bias-correction
        velocity_theta /= (1-pow(beta_velocity,i+1))
        s_theta /= (1-pow(beta_rmsprop,i+1))

        # update weights
        theta -= alpha_theta*velocity_theta/(np.sqrt(s_theta)+rms_offset) # gradient descent

        lamb += alpha_lamb*gradient_lamb
        
        if lamb < 0.0: # to not allow g to be positive
            lamb = 0.0 

    solution = {}
    solution_found = True
    if np.isinf(best_feasible_primary):
        solution_found = False
    else:    
        # solution['candidate_solution'] = candidate_solution.tolist()
        solution['candidate_solution'] = candidate_solution
        solution['best_index'] = best_index
        solution['best_feasible_g'] = best_feasible_g
        solution['best_feasible_f'] = best_feasible_primary

    solution['solution_found'] = solution_found
    
    if store_values:
        solution['theta_vals'] = theta_vals
        solution['f_vals'] = f_vals
        solution['g_vals'] = g_vals
        solution['lamb_vals'] = lamb_vals
        solution['L_vals'] = L_vals
    return solution