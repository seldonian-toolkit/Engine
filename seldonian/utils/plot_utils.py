import numpy as np
import matplotlib.pyplot as plt

def plot_gradient_descent(solution,primary_objective_name,save=False,savename='test.png'):
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
    fig = plt.figure(figsize=(12,4))
    fontsize=12
    solution_found = solution['solution_found']
    theta_vals = solution['theta_vals'] # includes intercept
    lamb_vals = solution['lamb_vals']
    f_vals = solution['f_vals']
    g_vals = solution['g_vals']
    L_vals = solution['L_vals']
    if solution_found:
        best_index = solution['best_index']
        best_feasible_f = solution['best_feasible_f']
        best_feasible_g = solution['best_feasible_g']

    ax_lamb = fig.add_subplot(1,4,1)
    ax_lamb.plot(np.arange(len(lamb_vals)),[x[0][0] for x in lamb_vals],
        linewidth=2)
    ax_lamb.set_xlabel("Iteration")
    ax_lamb.set_ylabel(r"$\lambda$",fontsize=fontsize)
    if solution_found:
        ax_lamb.axvline(x=best_index,linestyle='--',color='k')

    ax_f = fig.add_subplot(1,4,2)
    ax_f.plot(np.arange(len(f_vals)),f_vals,linewidth=2)
    ax_f.set_xlabel("Iteration")
    ax_f.set_ylabel(rf"$f(\theta)$: {primary_objective_name}",fontsize=fontsize)
    if solution_found:
        ax_f.axvline(x=best_index,linestyle='--',color='k')
        ax_f.axhline(y=best_feasible_f,linestyle='--',color='k')

    ax_g = fig.add_subplot(1,4,3)
    ax_g.plot(np.arange(len(g_vals)),[x[0][0] for x in g_vals],linewidth=2)
    ax_g.set_xlabel("Iteration")
    ax_g.set_ylabel(r"$g(\theta)$",fontsize=fontsize)
    ax_g.fill_between(np.arange(len(f_vals)),0.0,1e6,color='r',zorder=0,alpha=0.5)
    ax_g.set_ylim(min(-0.5,min(g_vals)),max(0.5,max(g_vals)*1.2))
    if solution_found:
        ax_g.axvline(x=best_index,linestyle='--',color='k')
        ax_g.axhline(y=best_feasible_g,linestyle='--',color='k')

    ax_L = fig.add_subplot(1,4,4)
    ax_L.plot(np.arange(len(L_vals)),L_vals,linewidth=2)
    ax_L.set_xlabel("Iteration")
    ax_L.set_ylabel(r"$L(\theta,\lambda)$",fontsize=fontsize)
    if solution_found:
        ax_L.axvline(x=best_index,linestyle='--',color='k')
    
    title = r"KKT optimization for $L(\theta,\lambda) = f(\theta) + \lambda g(\theta) $"
    plt.suptitle(title)
    plt.tight_layout()
    if save:
        plt.savefig(savename,dpi=300)
        print(f"Saved {savename}")