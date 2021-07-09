# Name: Orion Assefaw     Student ID: 201530497  

import numpy as np
from random import random
import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns


class MLP(object):

    def __init__(self, inputX, actual_y, seed = 3, eta = 0.15, xavier = True):
        # Hyperparameters and Network Architecture
        self.X = inputX
        self.y = actual_y
        self.randomSeed = seed
        self.learningRate = eta
        self.inputNodes = 2
        self.hiddenNodes = 4 # 4 of them in 1 hidden layer
        self.outputNodes = 1
        
        np.random.seed(self.randomSeed)
        
        if xavier:
            # Weights
            self.weights1 = np.random.rand(self.inputNodes,self.hiddenNodes) / np.sqrt(self.inputNodes)       
            self.weights2 = np.random.rand(self.hiddenNodes,self.outputNodes)/ np.sqrt(self.inputNodes)
            # Biases
            self.bias1 = np.zeros(4)
            self.bias1 = self.bias1.reshape(4,1)
            self.bias2 = np.zeros(1)
            self.bias2 = self.bias2.reshape(1,1)

        else:
            # Weights
            self.weights1 = np.random.rand(self.inputNodes,self.hiddenNodes)
            self.weights2 = np.random.rand(self.hiddenNodes,self.outputNodes)
            # Biases
            self.bias1 = np.random.rand(self.hiddenNodes,1)        
            self.bias2 = np.random.rand(self.outputNodes,1)    
      
    def forwardPropagation(self, X, sigmoid = True):
        # 1. Input Layer-to-Hidden Layer

        # Induced Local field (summation of (Input(X)*weights1) + bias1)
        self.v1 = np.dot(X,self.weights1) + self.bias1.T
        # Activation of the induced local field
        if sigmoid:
            self.a1 = self.sigmoid(self.v1)
        else:
            self.a1 = self.tanh(self.v1)
                
        # 2. Hidden Layer-to-Output Layer

        # Induced Local field (summation of (a1*weights2) + bias2 )
        self.v2 = np.dot(self.a1,self.weights2) + self.bias2.T
        # Activation of the induced local field
        if sigmoid:
            self.a2 = self.sigmoid(self.v2)
        else:
            self.a2 = self.tanh(self.v2)        
        
        return self.a2
    
    def backwardPropagation(self, X, y, sigmoid = True):
        
        # 1. Output Layer

        #  e(n) = d(n) - y(n)
        self.error_a2 = self.a2 - y        
        
        if sigmoid:
            # delta = e * sigmoid_derivative(a2)
            self.delta_a2 = self.error_a2 * self.sigmoid(self.a2, dev= True)
        else:
            # delta = e * tanh_derivative(a2)
            self.delta_a2 = self.error_a2 * self.tanh(self.a2, dev= True)

                
        # 2. Hidden Layer

        # error = summation(delta*weight)
        self.error_a1 = np.dot(self.delta_a2,self.weights2.T)
        if sigmoid:
            # delta = sigmoid_derivative(a1) * summation(delta*weight)  
            self.delta_a1 = self.sigmoid(self.a1, dev = True) * self.error_a1
        else:
            # delta = tanh_derivative(a1) * summation(delta*weight)
            self.delta_a1 = self.tanh(self.a1, dev = True) * self.error_a1      
        
               
        #Updating Weights

        # New weight = old weight - learning rate * summation(previous layer input * present layer delta)
        
        self.weights2 -= self.learningRate * np.dot(self.a1.T,self.delta_a2)
        self.weights1 -= self.learningRate * np.dot(X.T,self.delta_a1)
      
        self.bias2 -=  np.sum(self.learningRate * self.delta_a2)
        
        self.bias1 -= np.sum(self.learningRate * self.delta_a1)
        
        loss = np.sum(((self.error_a2)**2) / 2.0)
        
        return loss

    
    def fit(self,X,y, sigmoid = True):
        if sigmoid:
            yPredicted = self.forwardPropagation(X)
            self.backwardPropagation(X, y)
            return yPredicted
        else:
            yPredicted = self.forwardPropagation(X, False)
            self.backwardPropagation(X, y, False)
            #print(self.backwardPropagation(X, y, False)[1])
            return yPredicted
   
    
    def sigmoid(self, Z, dev = False):
        if dev == True:
            return Z * (1-Z)
        return 1 / (1 + np.exp((-1) * Z))

    def tanh(self, z, dev = False):
        t=(np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        if dev == True:
            return 1-t**2
        return t

    def separate(self, inputsample):
        inputsample = np.transpose(inputsample)

        if self.forwardPropagation(inputsample) < 0.5:
            return 0
        else:
            return 1

    def plot(self, h = 0.01, XOR = True):
        # plot properties 
        sns.set_style('darkgrid')
        plt.figure(figsize=(20, 20))

        plt.axis('scaled')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)

        colors = {
            0: "ro",
            1: "bo"
        }

        # plotting the four datapoints
        for i in range(len(self.X)):
            plt.plot([self.X[i][0]], [self.X[i][1]], colors[self.y[i][0]], markersize=20)

        xvalues = np.arange(-0.1, 1.1, h)
        yvalues = np.arange(-0.1, 1.1, h)

        # creating decision boundary
        xx, yy = np.meshgrid(xvalues, yvalues, indexing='ij')
        formarray = np.array([[self.separate([x, y]) for x in xvalues] for y in yvalues])

        # using the contourf function to create the plot
        plt.contourf(xx, yy, formarray, colors=['red', 'blue', 'blue', 'green'], alpha=0.4)
        if XOR:
            plt.title('XOR Decision Boundaries')
        else:
            plt.title('XNOR Decision Boundaries')
        plt.show()

  
# XOR
XOR_X = np.array([[0,0],[0,1],[1,0],[1,1]])
XOR_y = np.array([[0],[1],[1],[0]])


XOR_mlp = MLP(XOR_X, XOR_y, seed = 16, xavier = False)
XOR_errorsList = []
for i in range(7500):
    yPredicted = XOR_mlp.fit(XOR_X, XOR_y, sigmoid = True)
    if i % 100 == 0:
        print("XOR Result for iteration:" ,i ,"\n", np.round(yPredicted))
        theLoss = XOR_mlp.backwardPropagation(XOR_X, XOR_y, sigmoid = True)
        print("Iteration",i,"XOR loss:",theLoss) 
        XOR_errorsList.append(theLoss)
XOR_mlp.plot()

XOR_epochs = range(1, len(XOR_errorsList) + 1)
plt.plot(XOR_epochs, XOR_errorsList, 'bo')
plt.xlabel('Iterations(in hundreds)')
plt.ylabel('Loss values')
plt.title('Loss encountered during Training for XOR problem')
plt.show()


# XNOR
XNOR_X = np.array([[0,0],[0,1],[1,0],[1,1]])
XNOR_y = np.array([[1],[0],[0],[1]])


XNOR_mlp = MLP (XNOR_X, XNOR_y, seed = 16, xavier = False)

XNOR_errorsList = []
for i in range(7500):
    yPredicted2 = XNOR_mlp.fit(XNOR_X,XNOR_y, sigmoid = True)

    if i % 100 == 0:
        print("XNOR Result for iteration:" ,i ,"\n", np.round(yPredicted2))
        XNORLoss = XNOR_mlp.backwardPropagation(XNOR_X,XNOR_y, sigmoid = True)
        print("Iteration",i,"XNOR loss:",XNORLoss) 
        XNOR_errorsList.append(XNORLoss)
XNOR_mlp.plot(XOR = False)

XNOR_epochs = range(1, len(XNOR_errorsList) + 1)
plt.plot(XNOR_epochs, XNOR_errorsList, 'bo')
plt.xlabel('Iterations(in hundreds)')
plt.ylabel('Loss values')
plt.title('Loss encountered during Training for XNOR problem')
plt.show()
