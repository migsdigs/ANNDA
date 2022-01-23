from operator import concat
import numpy as np
import matplotlib as plt
import scipy

def data_function(classA_mean, classB_mean, classA_cov, classB_cov, n):
    
    classA = np.random.multivariate_normal(classA_mean,classA_cov, n).T
    classB = np.random.multivariate_normal(classB_mean,classB_cov, n).T

    classA_bias = np.ones((1,n), dtype=int)
    classB_bias = np.ones((1,n), dtype=int)

    classA = np.concatenate((classA, classA_bias), axis=0)
    classB = np.concatenate((classB, classB_bias), axis=0)

    classA_label = np.ones((1,n), dtype=int)
    classB_label = np.zeros((1,n), dtype=int)

    classA = np.concatenate((classA, classA_label), axis=0)
    classB = np.concatenate((classB, classB_label), axis=0)

    random_indices = np.random.permutation(2*n)
    data_preshuffle = np.concatenate((classA, classB), axis = 1)
    data = data_preshuffle[:,random_indices]

    return data, classA, classB

data_points, classA, classB = data_function(classA_mean=[5,5], classB_mean=[-5,-5], classA_cov=[[1.5,0], [0, 1.5]], classB_cov=[[1.5,0],[0, 1.5]], n=100)
print(np.shape(data_points))