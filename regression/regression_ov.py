import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def computeCost(x,y,theta):
	temp = np.dot(x,theta) - y
	return np.sum(np.power(temp,2))/(2*m)

def gradientDescent(x,y,theta,alpha,num_iter):
	for _ in range(num_iter):
		temp = np.dot(x,theta) - y
		temp = np.dot(x.T,temp)
		theta = theta - (alpha/m)*temp
	return theta

data = pd.read_csv('ex1data1.txt',header=None)
x = data.iloc[:,0]
y = data.iloc[:,1]
m = len(y)

x = x[:,np.newaxis]
y = y[:,np.newaxis]

alpha = 0.01
num_iter = 1500
theta = np.zeros((2,1))

ones = np.ones((m,1))
X = np.hstack((ones,x))
theta = gradientDescent(X,y,theta,alpha,num_iter)

plt.figure()
plt.scatter(x,y,color='#ff0000',marker='o')
plt.plot(x,np.dot(X,theta),color='#00ff00')
plt.title('Linear Regression',fontsize=24)
plt.xlabel('$x$',fontsize=24)
plt.ylabel('$y$',fontsize=24)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show() 