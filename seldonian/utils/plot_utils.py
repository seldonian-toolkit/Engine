import numpy as np
import matplotlib.pyplot as plt

def plot_gradient_descent(
	solution,
	primary_objective_name,
	save=False,
	savename='test.png',
	show=True):
	""" Make figure showing evolution of gradient descent.
	One row per constraint. The primary objective and lagrangian
	subplots are repeated in each row because they are not
	changing with the constraint

	Plots:
	i) primary objective
	ii) lagrange multipliers for each constraint, lambda_i
	iii) each constraint function, g_i
	iv) Lagranian L = f + sum_i^n {lambda_i*g_i}

	:param solution: The solution dictionary returned by gradient descent
	:type solution: dict
	:param primary_objective_name: The label you want displayed on the plot
		for the primary objective
	:type primary_objective_name: str
	:param save: Whether to save the plot
	:type save: bool
	:param savename: The full path where you want to save the plot
	:type savename: str
	:param show: Whether to show the plot with plt.show().
		Only relevant when save=False
	:type show: bool
	"""
	fontsize=10
	theta_vals = solution['theta_vals'] 
	lamb_vals = solution['lamb_vals'] # i x j array where i is number of iterations, j is number of constraints
	f_vals = solution['f_vals'] # length = i array where i is number of iterations 
	g_vals = solution['g_vals'] # i x j array where i is number of iterations, j is number of constraints
	L_vals = solution['L_vals']
	
	best_index = solution['best_index']
	best_f = solution['best_f']
	best_g = solution['best_g']

	if 'constraint_strs' in solution.keys():
		constraint_strs = solution['constraint_strs']
	else:
		constraint_strs = []

	n_constraints = g_vals[0].shape[0]
	n_cols = 4 

	fig = plt.figure(figsize=(10,4+(n_constraints-1)*2),
		constrained_layout=True)

	# create n_constraints x 1 subfigs
	subfigs = fig.subfigures(nrows=n_constraints, ncols=1)
	
	if not isinstance(subfigs,np.ndarray):
		subfigs = np.array([subfigs])
	
	title=rf"KKT optimization for $L(\theta,\lambda) = \hat{{f}}(\theta,D_\mathrm{{cand}}) + \sum_{{k=1}}^{{{n_constraints}}}{{\lambda_k}} \mathrm{{HCUB}}(\hat{{g}}_k(\theta,D_\mathrm{{cand}})) $"	
	fig.suptitle(title)
	
	# 1 row per constraint, f and L subplots repeated in each row
	for constraint_index, subfig in enumerate(subfigs):
		if len(constraint_strs) > 0:
			c_str = constraint_strs[constraint_index]
			subfig.suptitle(rf'$g_{constraint_index+1}$: {c_str} $\leq 0$')
		row_number = constraint_index + 1 # 1,2,3,...

		# create 1x4 subplots per subfig
		axs = subfig.subplots(nrows=1, ncols=4)
		for col, ax in enumerate(axs):
			if col == 0:
				# Primary objective, same for each constraint
				ax.plot(np.arange(len(f_vals)),f_vals,linewidth=2)
				ax.set_xlabel("Iteration")
				ax.set_ylabel(rf"$\hat{{f}}(\theta,D_\mathrm{{cand}})$: {primary_objective_name}",fontsize=fontsize)
				ax.axvline(x=best_index,linestyle='--',color='k')
				ax.axhline(y=best_f,linestyle='--',color='k')
			if col == 1:
				# lambda[constraint_index]
				lamb_vals_this_constraint = [x[constraint_index] for x in lamb_vals]
				ax.plot(np.arange(len(lamb_vals)),lamb_vals_this_constraint,
					linewidth=2)
				ax.set_xlabel("Iteration")
				ax.set_ylabel(rf"$\lambda_{row_number}$",fontsize=fontsize)
				ax.axvline(x=best_index,linestyle='--',color='k')
				best_lamb = lamb_vals[best_index][constraint_index]
				ax.axhline(y=best_lamb,linestyle='--',color='k')
			if col == 2:
				# g[constraint_index]
				g_vals_this_constraint = [x[constraint_index] for x in g_vals]
				ax.plot(np.arange(len(g_vals)),g_vals_this_constraint,linewidth=2)
				ax.set_xlabel("Iteration")
				ax.set_ylabel(rf"$\mathrm{{HCUB}}(\hat{{g}}_{{{row_number}}}(\theta,D_\mathrm{{cand}}))$",fontsize=fontsize)
				ax.fill_between(np.arange(len(g_vals)),0.0,1e6,color='r',zorder=0,alpha=0.5)
				ax.set_ylim(min(-0.25,min(g_vals_this_constraint)),max(0.25,max(g_vals_this_constraint)*1.2))
				ax.axvline(x=best_index,linestyle='--',color='k')
				ax.axhline(y=best_g[constraint_index],linestyle='--',color='k')
				ax.set_xlim(0,len(g_vals))
			if col == 3:
				# Lagrangian, same for each constraint
				ax.plot(np.arange(len(L_vals)),L_vals,linewidth=2)
				ax.set_xlabel("Iteration")
				ax.set_ylabel(r"$L(\theta,\lambda)$",fontsize=fontsize)
				ax.axvline(x=best_index,linestyle='--',color='k')
				best_L = L_vals[best_index]
				ax.axhline(y=best_L,linestyle='--',color='k')
	if save:
		plt.savefig(savename,dpi=300)
		print(f"Saved {savename}")
	else:
		if show:
			plt.show()
	return fig