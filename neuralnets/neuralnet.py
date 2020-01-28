import numpy as np

from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import rcParams

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1.0-x)

class NeuralNetwork():

    def __init__(self,x,y):
        self.input = x
        self.weights1 = np.random.randn(self.input.shape[1],4)
        self.weights2 = np.random.randn(4,1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        dweights2 = np.dot(self.layer1.T, (2*(self.y - self.output)*sigmoid_derivative(self.output)))
        dweights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output)*sigmoid_derivative(self.output),self.weights2.T)*sigmoid_derivative(self.layer1)))

        self.weights1 += dweights1
        self.weights2 += dweights2

if __name__ == "__main__":

    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    
    nn = NeuralNetwork(X,y)

    for i in range(1500):
        nn.feedforward()
        nn.backprop()

    print(nn.output)