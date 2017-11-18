import numpy as np
from testYa import hello


temp = []

userInput = input("Do you want to recall the saved weights?(y/n): ")


if userInput is "y" or userInput is "Y":
    with open("weights.txt", "r") as f:
        for w in f:
            temp.append(float(w.strip()))
    temp = np.asarray(temp).reshape((2, 3))
    print("Weights 1 =\n", temp)

    temp = []  # Bring the array bac to list

    with open("weights2.txt", "r") as f:
        for w in f:
            temp.append(float(w.strip()))
    temp = np.asarray(temp).reshape((3, 1))
    print("Weights 2 =\n", temp)



