# Imports
import numpy as np
import matplotlib as plt


# Neural Network Class
class NeuralNetwork:
    def __init__(self, architecture, lr, epochs, activation_func, activation_func_der, momentum, alpha):
        self.arch = architecture
        self.lr = lr
        self.epochs = epochs

        # Activation Functions and Respective Vectorized Forms
        self.activation_func = activation_func
        self.activation_func_der = activation_func_der
        self.act_functv = np.vectorize(self.activation_func)
        self.act_func_derv = np.vectorize(self.activation_func_der)

        self.momentum = momentum
        self.alpha = alpha

        self.n_layers = len(self.arch) - 1  # Number of layers

        # Forward Memory for Layer Outputs (H, O for 2 layer) - to be used in delta calculation
        self.forward_mem = [i for i in range(self.n_layers)]
        # Backward Memory for deltas of backprop - to be used in weight update
        self.backward_mem = [i for i in range(self.n_layers)]

        self.X  # Input Data
        self.T  # Output Data
        self.O  # Output of Network

    def weight_init(self):
        '''
        Generates Initialised Weight Matrices
        Weight Matrices of each layer is stores as an index
        Weight Matrix of first layer W[0]
        Weight Matrix of last layer W[n]
        '''
        self.weight_matrices = [np.random.normal(0,1, size=(self.arch[layer+1], self.arch[layer]+1)) for layer in range(self.n_layers)]    

    def momentum_init(self):
        '''
        Initialised Momentum Matrices
        Same dimensions as Weight Matrices
        '''
        self.momentum_matrices = [np.zeros((self.arch[layer+1], self.arch[layer]+1)) for layer in range(self.n_layers)]

    def forward_pass(self, X):
        '''
        Perform Forward Pass
        '''
        self.X = X
        input = self.X
        for layer in range(self.n_layers):
            # Add row of Ones to input
            input = np.append(input, [np.ones(np.shape(input)[1])], axis=0)

            # Multiply Layer Inputs by Weights
            output_pre_act = np.matmul(self.weight_matrices[layer],input)

            # Append Pre Activated Output to be Used in Backprop
            self.forward_mem.append(output_pre_act)

            # Apply Activation function to output
            output = self.act_functv(output_pre_act)

            input = output

        self.O = output

    def backward_pass(self, T):
        '''
        Perform Backward Pass
        '''
        self.T = T
        # Determine Generalised Error of Output Layer
        delta_o = np.multiply((self.O - self.T),self.act_func_derv(self.forward_mem[-1]))
        self.backward_mem[-1] = delta_o

        prev_delta = delta_o

        # Propagate backwards through the layers (from nth to 0th)
        for layer in range((self.n_layers-2), -1, -1):
            # (V^T * prev_delta) .* vfunc(prev_output*)
            # V^T * prev_delta
            vT_times_prev_delta = np.matmul(self.weight_matrices[layer].T, prev_delta)
            delta = np.multiply(vT_times_prev_delta, self.act_func_derv())



    def weight_update(self):
        return "yep"
