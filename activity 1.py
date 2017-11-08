
import matplotlib.pyplot as plt
import numpy as np


class Neural_Object(object):

    def __init__(self):
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        # Weights Matrix btw layers
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # propagate the inputs through the network
    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a1 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.z2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    # derivative of sigmoid function
    def sigmoidPrime(self, z):
        return np.exp(-z)/((1 + np.exp(-z))**2)

    # Compute cost for given X,y, use weights already stored in class.
    def costFunction(self, X, y):
        self.yHat = self.forward(X)
        J = 0.5 * sum((y - self.yHat) ** 2)
        return J

    # derivative of cost function respect to weight 2 n 1
    def costFunctionPrime(self, X, y):
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a1.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2










'''
plt.figure(1)
plt.subplot(311)
plt.plot([1, 2, 3, 4])
plt.xlabel('days')
plt.text(0.5, 2.5, r'$\mu=100, \sigma = 15$')
plt.xscale('log')
plt.xlim(0, 100)


plt.title('Line Graph 1/2')
plt.subplot(313)
plt.plot([4, 5, 6, 7])
plt.title('line Graph 2/2')

plt.show()
'''