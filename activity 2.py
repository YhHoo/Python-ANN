import numpy as np

temp = []
weight = np.arange(6).reshape((2, 3))
print(weight)

userInput = input("Do you want to save the weights?(y/n): ")

if userInput is "y" or "Y":
    with open("weights.txt", "w") as f:
        for w in weight.flatten():
            f.write(str(w) + "\n")
    print("Saved ! \n")

userInput = input("Do you want to recall the saved weights?(y/n): ")
if userInput is "y" or "Y":
    with open("weights.txt", "r") as f:
        for w in f:
            temp.append(int(w.strip()))
    temp = np.asarray(temp).reshape((2, 3))


print("Recalled: ", temp)
