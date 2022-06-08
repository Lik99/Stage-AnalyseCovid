import pandas as pd
import numpy as np
import cvxpy 
import scipy
import cvxopt 
import matplotlib.pyplot as plt


# Initialize the data
y = pd.read_csv('Classeur1.csv', sep=";")
y = y['cas'].to_numpy()
y = np.log(y)

# Initialize the parameters
n = y.size
ones_row = np.ones((1, n))
D = scipy.sparse.spdiags(np.vstack((ones_row, -2*ones_row, ones_row)), range(3), n-2, n)

# lambda = Paramètre régulateur pour notre problème
lambda_list = [0, 1000, 15000]

# H-P Trend Filtering - Smoother than L1
solver = cvxpy.CVXOPT
reg_norm = 2

# L1 Trend Filtering - Less linear than H-P
# solver = cvxpy.ECOS
# reg_norm = 1


fig, ax = plt.subplots(len(lambda_list)//3, 3, figsize=(20,20))
ax = ax.ravel()
ii = 0
for lambda_value in lambda_list:
    x = cvxpy.Variable(shape=n) 
    # x is the filtered trend that we initialize
    objective = cvxpy.Minimize(0.5 * cvxpy.sum_squares(y-x) 
                  + lambda_value * cvxpy.norm(D@x, reg_norm))
    # Note: D@x is syntax for matrix multiplication
    problem = cvxpy.Problem(objective)
    problem.solve(solver=solver, verbose=False)

    # Note : Arrangement de nos graphiques (Labels, titres, etc.)
    ax[ii].plot(np.arange(1, n+1), y, linewidth=1.0, c='b')
    ax[ii].plot(np.arange(1, n+1), np.array(x.value), 'b-', linewidth=1.0, c='r')
    ax[ii].set_xlabel('Time')
    ax[ii].set_ylabel('Covid Cases')
    ax[ii].set_title('Lambda: {}\nSolver: {}\nObjective Value: {}'.format(lambda_value, problem.status, round(objective.value, 3)))
    ii+=1
    
plt.tight_layout()
plt.savefig('results/trend_filtering_L{}.png'.format(reg_norm))
plt.show()