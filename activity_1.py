
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
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    # helper function of getParams(), setParams() and unrollGradients()
    def getParams(self):
        # Unroll the current weight matrix and save into a list
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        W1_start = 0
        W1_end = self.inputLayerSize * self.hiddenLayerSize
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def unrollGradient(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


class Trainer(object):
    def __init__(self, N):
        # make a reference to the ANN object
        self.N = N
        self.J = []

    # update the network weights and the append the cost to J list
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))  # MODIFIED

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.unrollGradient(X, y)

        return cost, grad

    def train(self, X, y):
        # Make an internal variable for the callback function:
        self.X = X
        self.y = y

        # Make empty list to store costs:
        self.J = []

        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS',
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res


NN = Neural_Object()
T = Trainer(NN)
T.train(X, y)

scorePrediction = NN.forward(X)
print('\nPredicted Score =\n', scorePrediction)
print('\nIdeal Score =\n', y)

# showing graphs of cost vs iterations
plt.plot(T.J)
plt.grid(True)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

print("\nOptimized Weights 1:\n", NN.W1)
print("\nOptimized Weights 2:\n", NN.W2)


userInput = input("Do you want to save the weights?(y/n): ")

if userInput is 'y' or userInput is 'Y':
    with open("weights.txt", "w") as f:
        for w in NN.W1.flatten():
            f.write(str(w) + "\n")
    with open("weights2.txt", "w") as f:
        for w in NN.W2.flatten():
            f.write(str(w) + "\n")
    print("Saved ! \n")
else:
    print("Weights Discarded !\n")























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