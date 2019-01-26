import numpy as np
import mnist

x_train, y_train, x_test, y_test = mnist.load()


class SimpleNeuralNetwork:
    num_inputs = 0
    weights = np.array([])
    learning_rate = 0

    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = np.zeros((1, num_inputs))

    def compute(self, x):
        input_vectorized = np.array(x)
        linear_combination = np.dot(input_vectorized, self.weights)
        return 1/(1+np.exp(-linear_combination))

    def learn(self, x, t):



derp = SimpleNeuralNetwork(785)
