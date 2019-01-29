import numpy as np
import mnist
from matplotlib import pyplot as plt
x_train, y_train, x_test, y_test = mnist.load()

class SimpleNeuralNetwork:

    def __init__(self, size):
        self.weights = 0.001*np.random.randn(size, )
        self.bias = np.random.rand(1,)
        self.alpha = 0.01

    def compute(self, input_set):
        return np.ndarray.flatten(self.sigmoid(-np.matmul(input_set, self.weights)))

    def log_reg(self, training_input, desired_output):
        error = desired_output - self.compute(training_input)
        self.weights -= self.alpha*np.matmul(error, training_input)


    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(x))


derp = SimpleNeuralNetwork(784)
x_mini = x_train[:10]
y_mini = y_train[:10]
print(derp.log_reg(x_mini, y_mini))
