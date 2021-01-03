## GRADIENT DESCENT WITH BACKTRACKING LINE SEARCH
#
# Implements gradient descent for convex quadratic functions
# with backtracking line search to select the step size
#
# Author: Abijith J Kamath
# https://kamath-abhijith.github.io

# %%
import numpy as np
import dill
import pickle

# %% FUNCTION DEFINITIONS
def backtracking(x, f, grad, rho=0.5, beta=0.8):
    grad_x = np.array(grad(x.tolist()), dtype=np.float64)
    fx = np.array(f(x.tolist()), dtype=np.float64)

    step_size = 1.0
    while f((x-step_size*grad_x).tolist()) >= (fx-rho*step_size*np.linalg.norm(grad_x,2)**2):
        step_size *= beta

    return step_size

def grad_descent(x_init, f, grad, rho=0.5, beta=0.8, tol=1e-6, max_iterations=100):

    x = x_init
    for iter in range(max_iterations):
        grad_x = np.array(grad(x.tolist()), dtype=np.float64)
        step_size = backtracking(x, f, grad, rho, beta)
        # step_size = 0.1

        x -= step_size*grad_x
        if np.linalg.norm(grad_x,2) <= tol:
            break

    return iter+1, x

# %% READ FUNCTIONS
fx = dill.loads(pickle.load(open('f6.pkl','rb')))
grad_fx = dill.loads(pickle.load(open('grad_f6.pkl','rb')))

# %% MINIMISATION OF f6 USING GRADIENT DESCENT

tol = 1e-7
max_iterations = 1000
x_init = np.array([10, 100, 100, 10], dtype=np.float64)
alpha = 0.5
beta = 0.5

num_iter, x_gd = grad_descent(x_init, fx, grad_fx, alpha, beta, tol, max_iterations)

print("Gradient descent with exact step size takes %d iterations to solve for the minimiser with an error of %.7f"%(num_iter, tol))
print("The minimum value is %.2f"%(fx(x_gd.tolist())))
print("The minimiser is", x_gd)

# %%
