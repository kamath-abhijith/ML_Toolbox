
# %% 
import numpy as np
import dill
import pickle

# %% FUNCTION DEFINITIONS

def grad_p1(x):
    grad = np.array([-400*x[0]*(x[0]-x[1]**2) - 2*(1-x[0]), 200*(x[0]-x[1]**2)])
    return grad

def hess_p1(x):
    hess = np.array([[-400*(x[1]-x[0]**2) + 800*x[0]**2 + 2, -400*x[0]],
              [-400*x[0], 200]])
    return hess

def grad_p4(x):
    Q = np.array([[4,2],
                  [2,2]])
    b = np.array([1,-1])
    return Q.dot(x) + b

def newton_descent(x_init, grad, hess, tol=1e-6, max_iterations=100):
    x = x_init
    for iter in range(max_iterations):
        gradient = grad(x)
        hessian = hess(x)
        x -= np.linalg.inv(hessian).dot(gradient)
        if np.linalg.norm(gradient,2) <= tol:
            break

    return iter+1, x

def quasi_newton(x_init, hess_init, grad, tol=1e-6, max_iterations=100):
    x = x_init
    B_DFP = hess_init
    for iter in range(max_iterations):
        xprev = x.copy()
        grad_x = grad(x)

        print(B_DFP)

        x -= B_DFP.dot(grad_x)

        gradnext_x = grad(x)
        delta = x-xprev
        gamma = gradnext_x-grad_x
        B_DFP = B_DFP + np.outer(delta,delta)/(delta.dot(gamma)) - np.outer(B_DFP.dot(gamma), B_DFP.dot(gamma))/(gamma.dot(B_DFP.dot(gamma)))
        if np.linalg.norm(grad_x,2) <= tol:
            break
        
    return iter+1, x

def grad_descent(x_init, grad, tol=1e-6, max_iterations=100):

    x = x_init
    for iter in range(max_iterations):
        grad_x = grad(x)
        # step_size = backtracking(x, f, grad, rho, beta)
        step_size = 0.05

        x -= step_size*grad_x
        if np.linalg.norm(grad_x,2) <= tol:
            break

    return iter+1, x

def conj_gradient(x_init, A, b, tol=1e-6, max_iterations=100):
    x = x_init
    d = b - A.dot(x)
    r = b - A.dot(x)
    for iter in range(max_iterations):
        alpha = r.dot(r)/(d.dot(A.dot(d)))
        x += alpha*d

        rold = r.copy()
        r -= alpha*A.dot(d)
        beta = r.dot(r)/(rold.dot(rold))
        d = r + beta*d

        if np.linalg.norm((A.dot(x)-b),2) <= tol:
            break

    return iter+1, x

def backtracking(x, f, grad, hess, rho=0.5, beta=0.8):
    hess_x = np.array(hess(x.tolist()), dtype=np.float64)
    grad_x = np.array(grad(x.tolist()), dtype=np.float64)
    fx = np.array(f(x.tolist()), dtype=np.float64)

    step_size = 1.0
    while f((x-step_size*hess_x.dot(grad_x)).tolist()) >= (fx-rho*step_size*grad_x.dot(hess_x.dot(grad_x))):
        step_size *= beta

    return step_size

def newton_method(x_init, f, grad, hess, alpha=0.1, beta=0.7, tol=1e-6, max_iterations=100):
    x = x_init
    for iter in range(max_iterations):
        xold = x.copy()
        grad_x = np.array(grad(x.tolist()))
        hess_x = np.array(hess(x.tolist()))

        step_size = backtracking(x, f, grad, hess, alpha, beta)
        x -= step_size*hess_x.dot(grad_x)
        if np.linalg.norm(x-xold,2) <= tol:
            break

    return iter+1, x
# %% MAIN :: QUESTION 1

x_init = np.array([.0, .0])
tol = 1e-12
max_iterations = 2
num_iter, x_sol = newton_descent(x_init, grad_p1, hess_p1, tol, max_iterations)

print("Newton method takes %d iterations to solve within an error of %.4f"%(num_iter, tol))
print("The Newton method solution obtained: ", x_sol)

# x_init = np.array([1.0, 1.01])
# tol = 1e-12
# max_iterations = 1000
# num_iter, x_sol = grad_descent(x_init, grad_p1, tol, max_iterations)

# print("Gradient Descent takes %d iterations to solve within an error of %.4f"%(num_iter, tol))
# print("The Gradient Descent solution obtained: ", x_sol)

# %% MAIN :: QUESTION 4

x_init = np.array([-0.0,0.0])
hess_init = np.eye(2)
tol = 1e-12
max_iterations = 5
num_iter, x_sol = quasi_newton(x_init, hess_init, grad_p4, tol, max_iterations)

print("Rank-2 Quasi-Newton method takes %d iterations to solve within an error of %.4f"%(num_iter, tol))
print("The Rank-2 Quasi-Newton method solution obtained: ", x_sol)

# %% MAIN :: QUESTION 5

fx = dill.loads(pickle.load(open('f.pkl','rb')))
grad_fx = dill.loads(pickle.load(open('grad_f.pkl','rb')))
hess_fx = dill.loads(pickle.load(open('hessian_inv.pkl','rb')))

x_init = np.array([0.0,0.0])
tol = 1e-3
alpha = 0.1
beta = 0.7
max_iterations = 1000
num_iter, x_sol = newton_method(x_init, fx, grad_fx, hess_fx, alpha, beta, tol, max_iterations)

print("Newton Method with backtracking takes %d iterations to solve within an error of %.4f"%(num_iter, tol))
print("The Newton Method solution obtained: ", x_sol)

# %% MAIN :: QUESTION 6

x_init = np.array([0.0,0.0,0.0], dtype=np.float)
Q = np.array([[2,1,1],
             [1,2,1],
             [1,1,2]], dtype=np.float)
b = np.array([4,0,0], dtype=np.float)
tol = 1e-6
max_iterations = 10
num_iter, x_sol = conj_gradient(x_init, Q, b, tol, max_iterations)

print("Conjugate Gradient method takes %d iterations to solve within an error of %.4f"%(num_iter, tol))
print("The Conjugate Gradient solution obtained: ", x_sol)

# %% MAIN :: QUESTION 7

x_init = np.array([0,0,0,0], dtype=np.float)
Q = np.array([[1,2,-1,1],
             [2,5,0,2],
             [-1,0,6,0],
             [1,2,0,3]], dtype=np.float)
b = np.array([0,2,-1,1], dtype=np.float)
tol = 1e-6
max_iterations = 10
num_iter, x_sol = conj_gradient(x_init, Q, b, tol, max_iterations)

print("Conjugate Gradient method takes %d iterations to solve within an error of %.4f"%(num_iter, tol))
print("The Conjugate Gradient solution obtained: ", x_sol)
# %%
