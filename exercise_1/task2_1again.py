import numpy as np
import mnist
import time
from matplotlib import pyplot as plt

# import mnist data
x_train_raw, y_train_raw, x_test_raw, y_test_raw = mnist.load()
x_train_raw = np.c_[x_train_raw[:20000], np.ones(20000)]
y_train_raw = y_train_raw[:20000]
x_test_raw = np.c_[x_test_raw[-2000:], np.ones(2000)]
y_test_raw = y_test_raw[-2000:]

x_train = np.array([x for x, y in zip(x_train_raw, y_train_raw) if y in [2, 3]])
y_train = np.array([{2: 1, 3: 0}[y] for x, y in zip(x_train_raw, y_train_raw) if y in [2, 3]])
x_test = np.array([x for x, y in zip(x_test_raw, y_test_raw) if y in [2, 3]])
y_test = np.array([{2: 1, 3: 0}[y] for x, y in zip(x_test_raw, y_test_raw) if y in [2, 3]])


class Henk:

    def __init__(self, num_inputs):
        self.weights = 0.0000001*np.random.randn(num_inputs+1,)
        self.learning_rate = 0.0000001

    def calc_z(self, data_sets):
        return np.matmul(data_sets, self.weights)

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    def feed_forward(self, data_sets):
        return self.sigmoid(self.calc_z(data_sets))

    def gradient_descent(self, data_sets, desired_sets):
        z = self.calc_z(data_sets)
        y = self.sigmoid(z)
        error = desired_sets - y
        self.weights += self.learning_rate * np.matmul(error,data_sets)
        return error

    def loss(self, data_sets, desired_sets):
        z = self.calc_z(data_sets)
        y = self.sigmoid(z)
        t = desired_sets
        print("Y tho??", y)
        instantanious_loss = np.matmul(t, np.log(y)) + np.matmul((1.0-t), np.log(1.0-y))
        return -np.average(instantanious_loss)


def simple_wieght_plot(image):
    plt.imshow(image.reshape((28, 28)))
    plt.title('Weights')
    plt.colorbar()
    plt.show()

def print_np_info(data):
    print("~>", type(data), np.shape(data))

def get_random_images(x, y, batch_size):
    randints = np.random.randint(0, len(x), (batch_size,))
    return np.array([x[i] for i in randints]), np.array([y[i] for i in randints])

henk = Henk(784)
losses = []
for n in range(40):
    random_x, random_y = get_random_images(x_train, y_train, 20)
    error = henk.gradient_descent(random_x, random_y)
    ff = henk.feed_forward(x_test)-y_test
    sub_loss = henk.loss(x_test, y_test)
    losses.append(sub_loss)
simple_wieght_plot(henk.weights[:784])
plt.plot(losses)
plt.title('Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()
