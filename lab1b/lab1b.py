# Imports
import numpy as np
import matplotlib.pyplot as plt

# Auxiliary variables
colors = ['#1E90FF', '#FF69B4']

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

        self.input_mem = []

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
        self.weight_matrices = [np.random.normal(0, 1, size=(
            self.arch[layer+1], self.arch[layer]+1)) for layer in range(self.n_layers)]

    def momentum_init(self):
        '''
        Initialised Momentum Matrices
        Same dimensions as Weight Matrices
        '''
        self.momentum_matrices = [np.zeros(
            (self.arch[layer+1], self.arch[layer]+1)) for layer in range(self.n_layers)]

    def forward_pass(self, X):
        '''
        Perform Forward Pass
        '''
        self.X = X
        input = self.X
        self.input_mem = []
        self.input_mem.append(
            np.append(input, [np.ones(np.shape(input)[1])], axis=0))
        for layer in range(self.n_layers):
            # Add row of Ones to input
            input = np.append(input, [np.ones(np.shape(input)[1])], axis=0)

            # Multiply Layer Inputs by Weights (H*, O*) (7xN)
            output_pre_act = np.matmul(self.weight_matrices[layer], input)

            # Append Pre Activated Output to be Used in Backprop
            self.forward_mem.append(output_pre_act)

            # Apply Activation function to output
            output = self.act_functv(output_pre_act)
            self.input_mem.append(
                np.append(output, [np.ones(np.shape(output)[1])], axis=0))

            input = output

        self.O = output

        # Remove Outputs from Input Mem
        self.input_mem = self.input_mem[:-1]

    def backward_pass(self, T):
        '''
        Perform Backward Pass
        '''
        self.T = T
        # Determine Generalised Error of Output Layer
        # (1xN - 1xN) .* 1xN = 1xN
        delta_o = np.multiply(
            (self.O - self.T), self.act_func_derv(self.forward_mem[-1]))
        self.backward_mem[-1] = delta_o

        prev_delta = delta_o

        # Propagate backwards through the layers (from nth to 0th)
        for layer in range((self.n_layers-2), -1, -1):
            # (V^T * prev_delta) .* vfunc(prev_output*)
            # V^T * prev_delta
            vT_times_prev_delta = np.matmul(
                self.weight_matrices[layer].T, prev_delta)  # (8x1 * 1xN)

            # Derivative of Activation Function
            act_funcv = np.append(self.act_functv(self.forward_mem[layer]), [
                                  np.ones(np.shape(self.forward_mem[layer])[1])], axis=0)

            act_func_der = 0.5*(np.multiply((1 + act_funcv), (1 - act_funcv)))

            delta = np.multiply(vT_times_prev_delta,
                                act_func_der)              # 8xN .* 8xN
            # Remove Last Row that accounted for bias term
            delta = delta[:-1, :]

            # Store Delta in Backward Memory
            self.backward_mem[layer] = delta

            prev_delta = delta

    def weight_update(self):
        '''
        Perform Weight Update
        '''

        if not self.momentum:
            for layer in range(self.n_layers):
                delta_weights = self.lr * \
                    np.matmul(self.backward_mem[layer],
                              self.input_mem[layer].T)

        else:
            print("yep")
        self.weight_matrices[layer] = self.weight_matrices[layer] + \
            delta_weights


# Functions

# Data Generation
def gen_data_clusters(N, mean_A1, mean_A2, cov_A, mean_B, cov_B):
    # Class A
    X_A1 = np.random.multivariate_normal(mean_A1, cov_A, int(N/2)).T
    X_A2 = np.random.multivariate_normal(mean_A2, cov_A, int(N/2)).T
    X_A = np.append(X_A1, X_A2, axis=1)
    X_A = np.append(X_A, [1*np.ones(2*int(N/2))], axis=0)  # Class label

    # Class B
    X_B = np.random.multivariate_normal(mean_B, cov_B, N).T
    X_B = np.append(X_B, [-np.ones(2*int(N/2))], axis=0)  # Class label

    return X_A, X_B

# Subsampling of data


def subsample_mix_classes(X_A, X_B, f_A, f_B):
    # Subsample classes
    N_A, N_B = np.shape(X_A)[1], np.shape(X_B)[1]
    random_subs_indices_A = np.random.choice(
        N_A, size=int(N_A*f_A), replace=False)
    random_subs_indices_B = np.random.choice(
        N_B, size=int(N_B*f_B), replace=False)
    X_A_train = X_A[:, random_subs_indices_A]
    X_B_train = X_B[:, random_subs_indices_B]
    X_A_valid = X_A[:, [i for i in range(
        N_A) if i not in random_subs_indices_A]]
    X_B_valid = X_B[:, [i for i in range(
        N_B) if i not in random_subs_indices_B]]

    # Mix classes
    N_train = np.shape(X_A_train)[1] + np.shape(X_B_train)[1]
    N_valid = np.shape(X_A_valid)[1] + np.shape(X_B_valid)[1]
    random_col_indices_train = np.random.choice(
        N_train, size=N_train, replace=False)
    random_col_indices_valid = np.random.choice(
        N_valid, size=N_valid, replace=False)
    X_train = np.append(X_A_train, X_B_train, axis=1)[
        :, random_col_indices_train]
    X_valid = np.append(X_A_valid, X_B_valid, axis=1)[
        :, random_col_indices_valid]

    # Define labels vector
    T_train = X_train[-1, :]
    X_train = X_train[:-1, :]
    T_valid = X_valid[-1, :]
    X_valid = X_valid[:-1, :]

    return T_train, X_train, T_valid, X_valid


# Plotting Data Points
def plot_data(X, T):
    fig, ax = plt.subplots()
    ax.scatter(X[0, T > 0], X[1, T > 0], c=colors[0], label='Class B')
    ax.scatter(X[0, T < 0], X[1, T < 0], c=colors[1], label='Class A')
    ax.grid(visible=True)
    ax.legend()
    ax.set_title('Patterns and Labels')
    plt.show()
