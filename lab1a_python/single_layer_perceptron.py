# Generate Points to be used as data
from array import array
from tkinter import Variable
from data_for_binary_classification import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


## Task 1 - Single Layer Perceptron
# Number of Epochs
eta = 0.1
epoch = 100
n = 100

# Mode for sequential (0) or batch (1)
mode = 1

# Get data
data_points, classA, classB = data_function(classA_mean=[5,5], classB_mean=[2,2], classA_cov=[[0.5,0], [0, 0.5]], classB_cov=[[0.5,0],[0, 0.5]], n=n)

# Define weights Matrix
# w = np.array([[10.0],[-10.0],[0]]) 
w = np.array([np.random.multivariate_normal([0,0,0],[[1,0,0],[0,1,0],[0,0,1]])]).T
print(w)

plt.figure()
plt.scatter(classA[0,:], classA[1,:], c='b')
plt.scatter(classB[0,:], classB[1,:], c='r')


x = np.linspace(-10,10,1000)



# for each epoch loop
for i in range(epoch):
    dw = np.array([[0.0], [0.0], [0.0]])
    
    # Loop through each all samples
    for j in range(n*2):
        point = np.array([[data_points[0,j]], [data_points[1,j]], [1]])
        true_class = data_points[3,j]
        # print(true_class)

        # # PLR
        y = float(np.matmul(w.T, point))
        # print(y)

        if mode == 0:
            if int(true_class) == 1 and y < 0:
                w += eta*point
                # print("wrong")

            elif int(true_class) == 0 and y >= 0:
                w -= eta*point
                # print("wrong")

            else:
                # print("correct")
                pass

        else:
            if true_class == 1 and y < 0:
                dw += eta*point

            if true_class == 0 and y >= 0:
                dw -= eta*point
        
    w = w + dw

print(w)



# Plot line to separate classes

weight_gradient = float(w[0]/w[1])
line_gradient = -float(w[1]/w[0])
# print(weight_gradient)
print(line_gradient)

plt.grid()

line = line_gradient*x - w[2]/w[0]
plt.plot(x, line, c='k')
plt.legend(['Boundary','Class A', 'Class B'])
plt.show()