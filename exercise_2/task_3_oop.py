import numpy as np
import matplotlib.pyplot as plt
import mnist
import tqdm


def onehot_encode(Y, n_classes=10):
    onehot = np.zeros((Y.shape[0], n_classes))
    onehot[np.arange(0, Y.shape[0]), Y] = 1
    return onehot


def bias_trick(X):
    return np.concatenate((X, np.ones((len(X), 1))), axis=1)


def train_val_split(X, Y, val_percentage):
    dataset_size = X.shape[0]
    idx = np.arange(0, dataset_size)
    np.random.shuffle(idx)

    train_size = int(dataset_size*(1-val_percentage))
    idx_train = idx[:train_size]
    idx_val = idx[train_size:]
    X_train, Y_train = X[idx_train], Y[idx_train]
    X_val, Y_val = X[idx_val], Y[idx_val]
    return X_train, Y_train, X_val, Y_val


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return np.multiply(sigmoid(z), 1-sigmoid(z))


def tanh_sigmoid(z):
    return 1.7159*np.tanh((2.0/3.0)*z) + 0.01*z


def tanh_sigmoid_prime(z):
    return 1.7159*(2.0/3.0)*(1/((np.cosh(2.0/3.0*z))**2)) + 0.01


class Neuron_layer:

    def __init__(self, num_inputs, num_outputs, activation_func, activation_func_prime):
        # initialize weight matrix with normal distribution variables
        self.w = np.random.randn(num_outputs, num_inputs) * 1/np.sqrt(num_inputs)
        self.activation_func = activation_func
        self.activation_func_prime = activation_func_prime

    def z(self, x):
        return x.dot(self.w.T)

    def activation(self, x):
        return self.activation_func(self.z(x))


#mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()

# Pre-process data
X_train, X_test = (X_train/127.5)-1, (X_test/127.5)-1
X_train = bias_trick(X_train)
X_test = bias_trick(X_test)
Y_train, Y_test = onehot_encode(Y_train), onehot_encode(Y_test)

X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, 0.1)