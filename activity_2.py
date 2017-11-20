import numpy as np
import matplotlib.pyplot as plt
from activity_1 import Neural_Object
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

temp = []  # to store the weights

# ------------------------------------------
# Load the saved optimized weights from .txt
# ------------------------------------------

userInput = input("Do you want to recall the saved weights?(y/n): ")

if userInput is "y" or userInput is "Y":
    with open("weights.txt", "r") as f:
        for w in f:
            temp.append(float(w.strip()))
    OptimizedWeight_1 = temp
    temp = np.asarray(temp).reshape((2, 3))
    print("Weights 1 =\n", temp)
    temp = []  # Bring the array bac to list

    with open("weights2.txt", "r") as f:
        for w in f:
            temp.append(float(w.strip()))
    OptimizedWeight_2 = temp
    temp = np.asarray(temp).reshape((3, 1))
    print("Weights 2 =\n", temp)


# ------------------------------------------
# Load the saved optimized weights from .txt
# ------------------------------------------

NN = Neural_Object()
NN.setParams(OptimizedWeight_1 + OptimizedWeight_2)
print("\nWeight 1:\n", NN.W1)
print("\nWeight 2:\n", NN.W2)

# create all possible data set to test the model
hourStudy = np.linspace(0, 10, 100)
hourSleep = np.linspace(0, 5, 100)
# Normalize
hourSleepNorm = hourSleep / 10
hourStudyNorm = hourStudy / 5

# create 2D data sets(x, y) of all combinations of hSleep and hStudy
a, b = np.meshgrid(hourSleepNorm, hourStudyNorm)
dataSet = np.zeros((a.size, 2))

# set all column 0 to the hourStudyNorm[10000]
dataSet[:, 0] = a.ravel()
dataSet[:, 1] = b.ravel()

allOutputs = NN.forward(dataSet)


# -------------------------------
# Display the result in 3D graphs
# -------------------------------

xx = np.dot(hourStudy.reshape(100, 1), np.ones((1, 100)))
yy = np.dot(hourSleep.reshape(100, 1), np.ones((1, 100))).T

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(xx, yy, 100*allOutputs.reshape(100, 100),
                       cmap=cm.jet)

ax.set_xlabel('Hours Sleep')
ax.set_ylabel('Hours Study')
ax.set_zlabel('Test Score')

plt.show()
