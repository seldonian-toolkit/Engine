import numpy as np
import matplotlib.pyplot as plt

def plot_gradient_descent(solution,primary_objective_name,
    save=False,
    savename='test.png',
    show=True):
    """ Make figure showing evolution of gradient descent.
    Shows the values of the Lagrange multipliers, primary objective,
    constraint functions and the Lagrangian L = f + dot(lambda,g).

    :param solution: The solution dictionary returned by gradient descent
    :type solution: dict

    :param primary_objective_name: The label you want displayed on the plot
        for the primary objective
    :type primary_objective_name: str

    :param save: Whether to save the plot
    :type save: bool

    :param savename: The full path where you want to save the plot
    :type savename: str
    """
    
    fontsize=10
    solution_found = solution['solution_found']
    theta_vals = solution['theta_vals'] 
    lamb_vals = solution['lamb_vals'] # i x j array where i is number of iterations, j is number of constraints
    f_vals = solution['f_vals'] # length = i array where i is number of iterations 
    g_vals = solution['g_vals'] # i x j array where i is number of iterations, j is number of constraints
    L_vals = solution['L_vals']
    if solution_found:
        best_index = solution['best_index']
        best_feasible_f = solution['best_feasible_f']
        best_feasible_g = solution['best_feasible_g']
    n_constraints = g_vals[0].shape[0]
    n_cols = 4 
    # fig = plt.figure(figsize=(10,4+(n_constraints-1)*2))
    fig,axes = plt.subplots(n_constraints,4,figsize=(10,4+(n_constraints-1)*2))
    # 1 row per constraint, f and L subplots repeated in each row
    for constraint_index in range(n_constraints): # 0,1,2,...
        row_number = constraint_index + 1 # 1,2,3,...

        if n_constraints > 1:
            axes_this_constraint = axes[constraint_index]
        else:
            axes_this_constraint = axes
        
        # Subplot: Primary objective, repeated for each row
        ax_f = axes_this_constraint[0]
        ax_f.plot(np.arange(len(f_vals)),f_vals,linewidth=2)
        ax_f.set_xlabel("Iteration")
        ax_f.set_ylabel(rf"$\hat{{f}}(\theta)$: {primary_objective_name}",fontsize=fontsize)
        if solution_found:
            ax_f.axvline(x=best_index,linestyle='--',color='k')
            ax_f.axhline(y=best_feasible_f,linestyle='--',color='k')

        # Subplot: lambda[constraint_index]
        ax_lamb = axes_this_constraint[1]
        lamb_vals_this_constraint = [x[constraint_index][0] for x in lamb_vals]
        ax_lamb.plot(np.arange(len(lamb_vals)),lamb_vals_this_constraint,
            linewidth=2)
        ax_lamb.set_xlabel("Iteration")
        ax_lamb.set_ylabel(rf"$\lambda_{row_number}$",fontsize=fontsize)
        if solution_found:
            ax_lamb.axvline(x=best_index,linestyle='--',color='k')
            best_feasible_lamb = lamb_vals[best_index][constraint_index]
            ax_lamb.axhline(y=best_feasible_lamb,linestyle='--',color='k')

        # Subplot: g[constraint_index]
        ax_g = axes_this_constraint[2]

        g_vals_this_constraint = [x[constraint_index][0] for x in g_vals]
        ax_g.plot(np.arange(len(g_vals)),g_vals_this_constraint,linewidth=2)
        ax_g.set_xlabel("Iteration")
        ax_g.set_ylabel(rf"$\mathrm{{HCUB}}(\hat{{g}}_{{{row_number}}})$",fontsize=fontsize)
        ax_g.fill_between(np.arange(len(g_vals)),0.0,1e6,color='r',zorder=0,alpha=0.5)
        ax_g.set_ylim(min(-0.25,min(g_vals_this_constraint)),max(0.25,max(g_vals_this_constraint)*1.2))
        if solution_found:
            ax_g.axvline(x=best_index,linestyle='--',color='k')
            ax_g.axhline(y=best_feasible_g[constraint_index],linestyle='--',color='k')
        ax_g.set_xlim(0,len(g_vals))

        # Subplot: Lagrangian, repeated for each row
        ax_L = axes_this_constraint[3]
        ax_L.plot(np.arange(len(L_vals)),L_vals,linewidth=2)
        ax_L.set_xlabel("Iteration")
        ax_L.set_ylabel(r"$L(\theta,\lambda)$",fontsize=fontsize)
        if solution_found:
            ax_L.axvline(x=best_index,linestyle='--',color='k')
            best_feasible_L = L_vals[best_index]
            ax_L.axhline(y=best_feasible_L,linestyle='--',color='k')
    
    title = rf"KKT optimization for $L(\theta,\lambda) = \hat{{f}}(\theta) + \sum_{{k=1}}^{{{n_constraints}}}{{\lambda_k}} \mathrm{{HCUB}}(\hat{{g}}_k(\theta)) $"
    plt.suptitle(title)
    plt.tight_layout()
    if save:
        plt.savefig(savename,dpi=300)
        print(f"Saved {savename}")
    else:
        if show:
            plt.show()
    return fig