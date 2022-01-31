# Imports
import numpy as np
import matplotlib as plt


# Neural Network Class
class NeuralNetwork:
    def __init__(self, architecture, lr, epochs, activation_func, activation_func_der, momentum, alpha):
        self.arch = architecture
        self.lr = lr
        self.epochs = epochs
        self.activation_func = activation_func
        self.activation_func_der = activation_func_der
        self.momentum = momentum
        self.alpha = alpha

        self.n_layers = len(self.arch) - 1  # Number of layers

        # Forward Memory for Layer Outputs (H, O for 2 layer) - to be used in delta calculation
        self.forward_mem = []
        # Backward Memory for deltas of backprop - to be used in weight update
        self.backward_mem = []

        self.X  # Input Data
        self.T  # Output Data
        self.O  # Output of Network

    def weight_init(self):
        return "yep"

    def forward_pass(self):
        return "yep"

    def backward_pass(self):
        return "yep"

    def weight_update(self):
        return "yep"
