### cmaes.py

### This script uses the CMA-ES algorithm to 
### maximize an objective function

# import argparse
import numpy as np

solution_filename = './solution_files/solution.txt'
func_eval_filename = './solution_files/f.txt'

# def objective(x1,x2):
# 	### The function we are trying to maximize
# 	return 2500 - (1 - x1)**2 - 100*(x2 - x1**2)**2


def minimize(N,lamb,initial_solution,objective):

	# Set up object to store function evals 
	func_evals = []
          
	np.random.seed(0)
	# Initialize hyperparameters
	xmean = initial_solution.reshape(N,1)
	# xmean = np.array([0.0,1.0]).reshape(N,1)  # objective variables initial point
	sigma = 0.5          # coordinate wise standard deviation (step size)
	stopfitness = 0.75  # stop if fitness < stopfitness (minimization). 
	stopeval = 5e4  # stop after stopeval number of function evaluations

	# Strategy parameter setting: Selection  
	mu = lamb/2
	weights = np.log(mu+1/2) - np.log(np.arange(1,np.floor(mu)+1))
	mu = int(np.floor(mu)) 
	weights = weights/sum(weights)
	weights = weights.reshape(len(weights),1)
	mueff = sum(weights)**2/sum(weights**2)

	# Strategy parameter setting: Adaptation
	cc = (4+mueff/N) / (N+4 + 2*mueff/N)
	cs = (mueff+2) / (N+mueff+5)
	c1 = 2 / ((N+1.3)**2+mueff)
	cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((N+2)**2+mueff))
	damps = 1 + 2*max(0, np.sqrt((mueff-1)/(N+1))-1) + cs

	# Initialize dynamic (internal) strategy parameters and constants
	pc = np.zeros((N,1)) # evolution path for C
	ps = np.zeros((N,1)) # evolution path for sigma
	B = np.identity(N) # B defines the coordinate system
	D = np.ones((N,1)) # D defines the scaling
	C = np.identity(N) # covariance matrix
	invsqrtC = np.identity(N) # C^(-1/2), also identity matrix to start
	eigeneval = 0 # track update of B and D
	chiN=N**0.5*(1-1/(4*N)+1/(21*N**2))
	
	# -------------------- Generation Loop --------------------------------
	counteval = 0 
	while counteval < stopeval:
		# Generate and evaluate lambda offspring
		arx = np.zeros((N,lamb))
		arfitness = np.zeros(lamb)

		for k in range(0,lamb):
			m2 = D*np.random.normal(loc=0,scale=1,size=(N,1))
			res = xmean + sigma*np.dot(B,m2)
			arx[:,k] = res[:,0]
			# Call function
			theta=arx[:,k]
			arfitness[k] = objective(theta) 
			counteval+=1
			# add function eval to list of all evals
			# func_evals.append(f'{arfitness[k]:.5f}')
		
		# sort arfitness and get indices
		arindex = np.argsort(arfitness) # smallest to largest fitnesses since we are minimizing
		arfitness = arfitness[arindex] # apply indices to sort arfitness
		# print(arfitness)
		# store old mean and set new mean using recombination
		xold = xmean
		xmean = np.dot(arx[:,arindex[0:mu]],weights)  

		# Cumulation: Update evolution paths
		ps = np.dot(
			(1-cs)*ps + np.sqrt(cs*(2-cs)*mueff)*invsqrtC,
			(xmean-xold) / sigma
		)
		psnorm = np.linalg.norm(ps,ord=2)
		exp = 2*counteval/lamb
		hsig = psnorm/np.sqrt(1-(1-cs)**exp)/chiN < 1.4 + 2/(N+1)
		pc = (1-cc)*pc + hsig * np.sqrt(cc*(2-cc)*mueff) * (xmean-xold) / sigma

		# Adapt covariance matrix C
		# print(arx[:,arindex[0:mu]])
		# print(arx[arindex[0:mu],:].shape)
		# print(np.tile(xold, (mu,1)).shape)
		# print(np.tile(xold, (1, mu)))
		# artmp = ((1/sigma) * (arx[arindex[0:mu],:] - np.tile(xold, (mu,1)))).T
		artmp = (1/sigma) * (arx[:,arindex[0:mu]]-np.tile(xold, (1, mu)))

		# print(artmp.shape)
		# print(np.diagflat(weights).shape)
		C = (1-c1-cmu) * C \
		+ c1 * (np.dot(pc,pc.T) + (1-hsig) * cc*(2-cc) * C) \
		+ cmu * np.dot(np.dot(artmp,np.diagflat(weights)),artmp.T)

		# Adapt step size sigma
		sigma = sigma * np.exp((cs/damps)*(psnorm/chiN - 1))

		if (counteval - eigeneval) > lamb/(c1+cmu)/N/10:
			eigeneval = counteval
			C = np.triu(C) + np.triu(C,1).T # enforce symmetry
			eigenvalues, B = np.linalg.eig(C)# eigen decomposition, B==normalized eigenvectors
			D = np.sqrt(eigenvalues).reshape(N,1)
			invsqrtC = np.dot(np.dot(B,np.diagflat(1/D)),B.T)

		# Stop if we achieve a fitness within our tolerance range 
		if (arfitness[0] <= stopfitness):
			print(f"Found solution in {counteval} iterations")
			
			# Write function evals to file
			# with open(func_eval_filename,'w') as outfile_func:
			# 	outfile_func.write('\n'.join(func_evals))
			# print(f"Saved {func_eval_filename} ")
			
			# Write solution vector to file
			# Can use position of best fitness from current population or current xmean.
			# Will use former since it corresponds to the position where the function was actually evaluated
			bestpos = arx[:,arindex[0]]
			# bestx1 = bestpos[0]
			# bestx2 = bestpos[1]
			return bestpos
			# with open(solution_filename,'w') as outfile_sol:
			# 	outfile_sol.write(f'{bestx1:.5f}, {bestx2:.5f}')
			# print(f"Saved {solution_filename} ")
			break

		# Stop if the condition number of the matrix becomes too large	
		if (max(D) > 1e7 * min(D)):
			print(f"Condition number exceeds threshold. Stopping.")
			break
	else:
		print(f"{stopeval} iterations expired. Returning best solution at this point ")
		return arx[:,arindex[0]]

