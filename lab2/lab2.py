import numpy as np
import matplotlib.pyplot as plt


class Network():
    def __init__(self, N_RBF, mu, sigma):
        self.N_RBF = N_RBF
        self.mu = mu
        self.sigma = sigma

    def RBF(self, x):
        x_size = len(x)

        phi = np.zeros((x_size, self.N_RBF))
        for i in range(self.N_RBF):
            phi[:, i] = np.exp(-((x-self.mu[i])**2)/(2*self.sigma[i]**2))

        return phi

    def lms(self, phi, f):
        w = np.transpose(phi).dot(f).dot(
            np.linalg.inv(np.dot(np.transpose(phi), phi)))

        return w

    def forward(self):
        return "yes"

    def update(self):
        "pls"
