"""
LINEAR REGRESSION IN ONE VARIABLE ON HOUSING DATA

Finds the best fit straight line to
the given data points

Cost: Euclidean error
Optimiser: Gradient descent
Author: Abijith J. Kamath
		kamath-abhijith.github.io

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def computeCost(x,y,theta):
	"""
	Computes the Euclidean norm of the error
	Input:  Feature matrix, x
		    Measurements, y
		    Parameters, theta
	Output: Squared Euclidean error
	"""	
	temp = np.dot(x,theta) - y
	return np.sum(np.power(temp,2))/(2*m)

def gradientDescent(x,y,theta,alpha,num_iter):
	"""
	Performs gradient descent on Euclidean cost
	Input:  Feature matrix, x
		    Measurements, y
		    Initial parameters, theta
		    Learning rate, alpha
		    Number of iterations, num_iter
	Output: Optimised parameters, theta
	"""	
	for _ in range(num_iter):
		temp = np.dot(x,theta) - y
		temp = np.dot(x.T,temp)
		theta = theta - (alpha/m)*temp
	return theta

# Read and arrange data
data = pd.read_csv('ex1data1.txt',header=None)
x = data.iloc[:,0]
y = data.iloc[:,1]
m = len(y)

x = x[:,np.newaxis]
y = y[:,np.newaxis]

# Initialise optimisation
alpha = 0.01
num_iter = 1500
theta = np.zeros((2,1))

# Optimisation via gradient descent
ones = np.ones((m,1))
X = np.hstack((ones,x))
theta = gradientDescent(X,y,theta,alpha,num_iter)

# Figures
plt.figure()
plt.scatter(x,y,color='#ff0000',marker='o')
plt.plot(x,np.dot(X,theta),color='#00ff00')
plt.title('Linear Regression',fontsize=24)
plt.xlabel('$x$',fontsize=24)
plt.ylabel('$y$',fontsize=24)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show() 