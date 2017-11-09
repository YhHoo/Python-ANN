
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)  # 3x2
y = np.array(([75], [82], [93]), dtype=float)  # 3x1

# Normalize
X = X/np.amax(X, axis=0)
y = y/100  # Max test score is 100


class Neural_Object(object):

    def __init__(self):
        self.inputLayerSize = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1


        # Weights Matrix btw layers
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)  # 2x3
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)  # 3x1

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # propagate the inputs through the network
    def forward(self, X):
        self.z2 = np.dot(X, self.W1)  # 3x3
        self.a2 = self.sigmoid(self.z2)  # 3x3
        self.z3 = np.dot(self.a2, self.W2)  # 3x1
        yHat = self.sigmoid(self.z3)  # 3x1
        return yHat

    # derivative of sigmoid function
    def sigmoidPrime(self, z):
        return np.exp(-z)/((1 + np.exp(-z))**2)

    # Compute cost for given X,y, use weights already stored in class.
    def costFunction(self, X, y):
        self.yHat = self.forward(X)
        # sum finds the norm of the column vectors of y and yHat
        cost = 0.5 * sum((y - self.yHat) ** 2)
        return cost

    # derivative of cost function respect to weight 2 n 1
    def costFunctionPrime(self, X, y):
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a1.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    def testHelperFunction(self):
        print("Weight Matrix 1 =\n", self.W1)
        print("Weight Matrix 2 =\n", self.W2)
        param = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        print("Concatenate =\n", param)
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(param[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        print("Reshaped W1 =\n", self.W1)


aTest = np.reshape([1, 2, 3], (3, 1))
bTest = np.reshape([2, 3, 4], (3, 1))

print("a= \n", aTest, "\nb= \n", bTest)
print("\n", sum(bTest - aTest))




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



NN = Neural_Object()
cost1 = NN.costFunction(X, y)
dJdW1, dJdW2 = NN.costFunctionPrime(X, y)
print("Cost 1 =", cost1, "\n")
print("dJdW1 =\n", dJdW1, "\n")
print("dJdW2 =\n", dJdW2, "\n")

'''