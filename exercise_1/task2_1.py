import numpy as np
import mnist
from matplotlib import pyplot as plt

# import mnist data
x_train_raw, y_train_raw, x_test_raw, y_test_raw = mnist.load()
# get only relevant training data and label all 2's with target value 1 and 3's as 0
x_train = [x for x, y in zip(x_train_raw, y_train_raw) if y in [2, 3]]
y_train = [{2: 1, 3: 0}[y] for x, y in zip(x_train_raw, y_train_raw) if y in [2, 3]]
x_test = [x for x, y in zip(x_test_raw, y_test_raw) if y in [2, 3]]
y_test = [{2: 1, 3: 0}[y] for x, y in zip(x_test_raw, y_test_raw) if y in [2, 3]]

# format data and add bias term to each image to go from 784 to 785 input nodes
TRAIN_NUM_SAMPLES, IMG_SIZE = np.shape(x_train)
TEST_NUM_SAMPLES, _ = np.shape(x_test)
x_test = np.c_[x_test, np.ones(TEST_NUM_SAMPLES)]
x_train = np.c_[x_train, np.ones(TRAIN_NUM_SAMPLES)]


class SimpleNeuralNetwork:

    def __init__(self, size):
        self.weights = 0.001*np.random.randn(size+1, )
        self.alpha = 0.01

    def compute(self, input_set):
        return np.ndarray.flatten(self.sigmoid(-np.matmul(input_set, self.weights)))

    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(x))

    def log_reg(self, training_input, desired_output):
        error = desired_output - self.compute(training_input)
        # print(error)
        self.weights -= self.alpha*np.matmul(error, training_input)
        # print(self.weights)
        return error


henk = SimpleNeuralNetwork(784)
x_mini = x_train[:1000]
y_mini = y_train[:1000]
#print(x_mini)
#print(y_mini)
x_chunks = [x_train[i:i+10] for i in range(0,2000,10)]
y_chunks = [y_train[i:i+10] for i in range(0,2000,10)]

for x, y in zip(x_chunks, y_chunks):
    henk.log_reg(x, y)
    print(np.average(abs(henk.compute(x_test)-y_test)))
