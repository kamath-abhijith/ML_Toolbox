## BISECTION METHOD ON ONE-DIMENSIONAL FUNCTION
#
# Implements bisection search on a one-dimensional function
#
# Author: Abijith J Kamath
# https://kamath-abhijith.github.io

# %%
import numpy as np

# %% FUNCTION DEFINITIONS
def foo(x):
    return np.exp(-x) - np.cos(x)

def fibo(n):
    if n<0:
        print("Incorrect input")
    elif n==0:
        return 1
    elif n==1:
        return 1
    else:
        return fibo(n-1)+fibo(n-2)

def bisect(f, interval, max_iterations=100, mode='GS', tol=1e-20):

    x_left = interval[0]
    x_right = interval[1]

    for k in range(max_iterations):
        if mode == 'GS':
            rho = (0.5*(1+np.sqrt(5))/(1+0.5*(1+np.sqrt(5))))
        elif mode == 'FS':
            rho = fibo(max_iterations-k)/fibo(max_iterations-k+1)
        
        d = (x_right-x_left)*rho
        xmin = x_right - d
        xmax = x_left + d
        if f(xmin)<f(xmax):
            x_right = xmax
        else:
            x_left = xmin
        
        if mode == 'GS':
            if 0.5*(x_right-x_left) < tol:
                break
    
    return 0.5*(x_right-x_left), 0.5*(x_right+x_left)

# %% MINIMISATION USING BISECTION SEARCH
max_iterations = 9
interval = (0,1)
tol, xminFS = bisect(foo, interval, max_iterations, 'FS')
_, xminGS = bisect(foo, interval, max_iterations+100, 'GS', tol)

print("The minimiser using FS is %.3f with tolerance %.3f"%(xminFS, tol))
print("The minimiser using GS is %.3f with tolerance %.3f"%(xminGS, _))
# %%
