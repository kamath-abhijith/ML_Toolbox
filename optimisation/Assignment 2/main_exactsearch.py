## GRADIENT DESCENT WITH EXACT LINE SEARCH
#
# Implements gradient descent for convex quadratic functions
# with exact line search to select the step size
#
# Author: Abijith J Kamath
# https://kamath-abhijith.github.io

# %%
import numpy as np
import dill
import pickle

# %% FUNCTION DEFINITIONS
def exact_line_search(x, grad_x, hess_x):
    return np.linalg.norm(grad_x,2)**2/(grad_x.dot(hess_x.dot(grad_x)))

def grad_descent(x_init, grad, hess, tol=1e-6, max_iterations=100):

    x = x_init
    for iter in range(max_iterations):
        grad_x = np.array(grad(x.tolist()), dtype=np.float64)
        step_size = exact_line_search(x,grad_x,hess)

        x -= step_size*grad_x
        if np.linalg.norm(grad_x,2) <= tol:
            break

    return iter+1, x

def quad_grad(x):
    return A.dot(x) - b


# %% READ FUNCTIONS
fx = dill.loads(pickle.load(open('f5.pkl','rb')))
grad_fx = dill.loads(pickle.load(open('grad_f5.pkl','rb')))
hess_fx = dill.loads(pickle.load(open('hess_f5.pkl','rb')))

# %% MINIMISATION OF f5 USING GRADIENT DESCENT

tol = 1e-6
max_iterations = 100
x_init = np.array([0, 10], dtype=np.float64)
hess = np.array(hess_fx())
num_iter, x_gd = grad_descent(x_init, grad_fx, hess, tol, max_iterations)

print("Gradient descent with exact step size takes %d iterations to solve for the minimiser with an error of %.6f"%(num_iter, tol))
print("The minimum value is %.2f"%(fx((x_gd[0],x_gd[1]))))
print("The minimiser is", x_gd)

# %% DEFINE QUADRATIC FUNCTION
A = np.array([[0.78, -0.02, -0.12, -0.14],
             [-0.02, 0.86, -0.04, 0.06],
             [-0.12, -0.04, 0.72, -0.08],
             [-0.14, 0.06, -0.08, 0.74]])
b = np.array([0.76, 0.08, 1.12, 0.68])

tol = 1e-6
max_iterations = 100
x_init = np.zeros(4)
num_iter, x_gd = grad_descent(x_init, quad_grad, A, tol, max_iterations)

print("Gradient descent with exact step size takes %d iterations to solve for the minimiser with an error of %.6f"%(num_iter, tol))
print("The minimum value is %.2f"%(0.5*x_gd.dot(A.dot(x_gd))-b.dot(x_gd)))
print("The minimiser is", x_gd)

# %%
