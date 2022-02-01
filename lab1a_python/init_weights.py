import numpy as np
import matplotlib.pyplot as plt

N  = 100
# Class A
mean_A_first, cov_A = (-1,0.3), [[0.2**2, 0], [0, 0.2**2]]
mean_A_second = (1,0.3)

X_A_first = np.random.multivariate_normal(mean_A_first, cov_A, round(N/2)).T
X_A_second = np.random.multivariate_normal(mean_A_second, cov_A, round(N/2)).T

X_A = np.append(X_A_first, X_A_second, axis=1)

X_A = np.append(X_A,[np.ones(2*round(N/2))],axis=0) # Account for bias term
X_A = np.append(X_A,[-1*np.ones(N)],axis=0) # Class label

# Class B
mean_B, cov_B = (0,-0.1), [[0.3**2, 0], [0, 0.3**2]]
X_B = np.random.multivariate_normal(mean_B, cov_B, N).T
X_B = np.append(X_B,[np.ones(N)],axis=0) # Account for bias term
X_B = np.append(X_B,[np.ones(N)],axis=0) # Class label


# plt.figure()
# plt.scatter(X_A[0,:], X_A[1,:], c='b')
# plt.scatter(X_B[0,:], X_B[1,:], c='r')
# plt.legend(['Class A', 'Class B'])
# plt.grid()
# plt.show()

print(np.ones(20))


