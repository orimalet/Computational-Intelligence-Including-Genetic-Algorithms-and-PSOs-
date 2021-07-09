import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

np.random.seed(4)
class SLP(object):
    def __init__(self, iter = 50, learningRateEta = 0.01):
        self.iterations = iter
        self.eta = learningRateEta 
        self.weights = None
        self.bias = None      
        self.activation = self.signFunction            

    def fit(self, X, y):
        self.X = X
        self.y = y
        nsamples = self.X.shape[0]
        nfeatures = self.X.shape[1]
        self.weights = np.zeros(nfeatures)
        self.bias = 0
        self.errors = []

        labelsList = []
        for i in self.y:
            if i > 0:
                i = 1
                labelsList.append(i)
            else:
                i = -1
                labelsList.append(i)
        yLabel = np.asarray(labelsList)   

        print("Starting weights:", self.weights)
        print("Starting bias:", self.bias)

        for epoch in range(self.iterations):
            errors = 0         
            for pos, x in enumerate(self.X): 
                inducedLocalField = np.dot(x,self.weights) + self.bias
                yPredicted = self.activation(inducedLocalField)

                # error = Desired output minus Predicted output
                error = yLabel[pos] - yPredicted
                if error != 0:
                    errors+=1
                    # w(n+1) = w(n) + learning rate * error * x(n)
                    self.weights += self.eta*error*x
                    # bias(n+1) = bias(n) + learning rate * error
                    self.bias += self.eta*error

            print("Epoch",epoch,"weights:", self.weights)
            print("Epoch",epoch,"bias:", self.bias)              

            self.errors.append(errors)

        return self.bias, self.weights

    # Provides threshhold for classification based on sign
    def signFunction(self,x):
        return np.where(x>0, 1, -1)

    # Predicts the class labels in binary classification case
    def predict(self, X):
        inducedLocalField = np.dot(X, self.weights) + self.bias
        y_pred = self.activation(inducedLocalField)
        return y_pred      

    # Binary Classification accuracy calculating function
    def accuracyCalc(self,ylabel,ypred):
        correct = 0
        for i in range(len(ylabel)):
            if ylabel[i] == ypred[i]:
                correct += 1
        return correct / float(len(ylabel)) * 100.0

   
        



# Iris Dataset


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()

X = df.iloc[0:100, [0, 2]].values
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
y = np.where(y == y[0], 1, -1)
ppn_iris = SLP(20)  
print("The final bias and weights of the Iris Dataset are:", ppn_iris.fit(X,y))
ypred = ppn_iris.predict(X)
print("Perceptron classification accuracy for the Iris dataset is:",ppn_iris.accuracyCalc(y,ypred))

print("\n")

plt.plot(range(0, len(ppn_iris.errors)), ppn_iris.errors, marker='o')
plt.ylim([-0.1, max(ppn_iris.errors) + 0.1])		
plt.xlabel('Epochs')
plt.xticks([1,5,10,15,20])
plt.ylabel('Number of errors')
plt.title('Classification Errors encountered during Training for Iris Data')
plt.show()

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='Versicolor')
plt.xlabel('sepal length in cm')
plt.ylabel('sepal width in cm')
plt.legend(loc='upper left')
plt.show()

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, algorithm, resolution=0.02):
    markers = ('s', 'x', 'o')
    colors = ('red', 'blue', 'lightgreen')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                           np.arange(y_min, y_max, resolution))
    Z = algorithm.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


plot_decision_regions(X, y, algorithm = ppn_iris)
plt.legend(loc='upper left')
plt.title('Decision boundry of the Iris Dataset')
plt.show()

print("\n")

# Synthetic Data

N = 100
sampledata1 = np.random.normal(8, 1, (N,2))
sampledata2 = np.random.multivariate_normal([2,16], [[1,0],[0,1]], N)
firsty = [-1 for i in range(100)]
secondy = [1 for i in range(100)]
labels = np.concatenate((firsty,secondy))

dataset = np.concatenate((sampledata1,sampledata2))
# training_D1 = dataset[0:80]
# test_D1 = dataset[80:100]
# training_D2 = dataset[100:180]
# test_D2 = dataset[180:200]

ppn_synthetic = SLP(20)  
print("The final bias and weights of the Synthetic Dataset are:", ppn_synthetic.fit(dataset,labels))
predictions = ppn_synthetic.predict(dataset)
print("Perceptron classification accuracy for the synthetic dataset is:",ppn_synthetic.accuracyCalc(labels,predictions))

print("\n")


plt.plot(range(0, len(ppn_synthetic.errors)), ppn_synthetic.errors, marker='o')
plt.ylim([-0.1, max(ppn_synthetic.errors) + 0.1])
plt.xlabel('Epochs')
plt.xticks([1,5,10,15,20])
plt.ylabel('Number of errors')
plt.title('Classification Errors encountered during Training of Synthetic Dataset')
plt.show()


plot_decision_regions(dataset, labels, algorithm = ppn_synthetic)
plt.legend(loc='upper right')
plt.title('Decision boundry of the synthetic Dataset')
plt.show()