import numpy as np
import mnist
from matplotlib import pyplot as plt
x_train, y_train, x_test, y_test = mnist.load()

class SimpleNeuralNetwork:

    def __init__(self, size):
        self.weights = np.random.rand(1, size)
        self.bias = np.random.rand(1)
        self.alpha = 0.01

    def compute(self, x):
        return self.sigmoid(np.dot(x, self.weights)+self.bias)

    def log_reg(self, training_inputs, training_outputs):
        self.weights -= self.alpha*np.matmul(training_inputs, training_outputs - np.matmul(self.weights, training_inputs))

    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))


derp = SimpleNeuralNetwork(784)
derp.log_reg(x_train[0:10], y_train[0:10])
