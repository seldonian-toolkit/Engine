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
    verbose=False):
    # initialize modeling parameters
    theta = theta_init
    lamb = lambda_init
    velocity_theta, velocity_lamb = 0.0,0.0
    s_theta, s_lamb = 0.0,0.0
    rms_offset = 1e-6 # small offset to make sure we don't take 1/sqrt(very small) in weight update

    best_feasible_primary = np.inf # minimizing f so want it to be lowest possible
    best_index = 0  

    # Define a function that returns gradients of LM loss function using Autograd

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
            if i % 50 == 0:
                print(f"Iteration {i}")

        primary_val = primary_objective(theta)
        g_val = upper_bound_function(theta)

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
        # print("here1")
        grad_primary_theta_val = grad_primary_theta(theta)
        # print("calculating d upper_bound_function / dtheta")

        gu_theta = grad_upper_bound_theta(theta)
        # print(f"theta: {theta}")
        # print(f"d upper_bound_function / dtheta = {gu_theta}")
        grad_secondary_theta_val = lamb*gu_theta
        # print("here3")
        gradient_theta = grad_primary_theta_val + grad_secondary_theta_val
        # input("next")
        gradient_lamb = g_val
        # print(theta,lamb,primary_val,g_val)
        # print(grad_primary_theta_val,grad_secondary_theta_val)
        # print()

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
        if verbose:
            print("NSF")
    else:
        if verbose:
            print("Solution found")
    
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