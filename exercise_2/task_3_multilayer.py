import numpy as np
import matplotlib.pyplot as plt
import mnist
import tqdm

def shuffle_training_sets(X, Y):
    dataset_size = X.shape[0]
    id = np.arange(0, dataset_size)
    np.random.shuffle(id)

    return X[id], Y[id]

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


def should_early_stop(validation_loss, num_steps=3):
    """
    Returns true if the validation loss increases
    or stays the same for num_steps.
    --
    validation_loss: List of floats
    num_steps: integer
    """
    if len(validation_loss) < num_steps+1:
        return False

    is_increasing = [validation_loss[i] <= validation_loss[i+1] for i in range(-num_steps-1, -1)]
    return sum(is_increasing) == len(is_increasing)



def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return np.multiply(sigmoid(z), 1-sigmoid(z))


def tanh_sigmoid(z):
    return 1.7159*np.tanh((2.0/3.0)*z) + 0.01*z


def tanh_sigmoid_prime(z):
    return 1.7159*(2.0/3.0)*(1/((np.cosh(2.0/3.0*z))**2)) + 0.01


def softmax(z):
    z_exp = np.exp(z)
    return z_exp / z_exp.sum(axis=1, keepdims=True)


def forwards(x, layers):

    # add input layer activation
    layer_activation = layers[0].update_activation(x)

    # add hidden layer activations
    for i in range(1, len(layers)-1):
        layer_activation = layers[i].update_activation(layer_activation)

    # add output layer activation
    layers[-1].update_activation(layer_activation)
    return layers


def dw(delta, w):
    return np.dot(delta, w)


def backpropagate(x, t, layers, learning_rate):
    layers = forwards(x, layers)
    outputs = layers[-1].a
    normalization_factor = x.shape[0] * t.shape[1]

    for i in range(len(layers)-1, -1, -1):
        if i == len(layers)-1:
            layers[i].delta = -(t-outputs)
        else:
            layers[i].delta = np.multiply(layers[i].activation_func_prime(layers[i].z), np.dot(layers[i+1].delta, layers[i+1].weights))

    for i in range(len(layers)):
        if i==0:
            layers[i].dw = layers[i].delta.T.dot(x) / normalization_factor
        else:
            layers[i].dw = layers[i].delta.T.dot(layers[i-1].a) / normalization_factor


    for i in range(len(layers)):
        layers[i].weights -= learning_rate*layers[i].dw

    return layers


def calculate_accuracy(X, targets, layers):
    """
    Calculated the accuracy of the network.
    ---
    X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
    targets: shape: [batch_size, num_classes] numpy vector. Targets/label of images
    w: shape: [num_classes, num_features] numpy vector. Weight from input->output
    --
    Returns float
    """
    output = forwards(X, layers)[-1].a
    predictions = output.argmax(axis=1)
    targets = targets.argmax(axis=1)
    return (predictions == targets).mean()


def cross_entropy_loss(X, targets, layers):
    """
    Computes the cross entropy loss given the input vector X and the target vector.
    ---
    X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
    targets: shape: [batch_size, num_classes] numpy vector. Targets/label of images
    w: shape: [num_classes, num_features] numpy vector. Weight from input->output
    --
    Returns float
    """
    output = forwards(X, layers)[-1].a
    assert output.shape == targets.shape
    log_y = np.log(output)
    cross_entropy = -targets * log_y
    return cross_entropy.mean()

class Layer:
    def __init__(self, input_size, output_size, activation_func, activation_func_prime):
        self.delta = 0
        self.weights = np.random.randn(output_size, input_size) * (1/np.sqrt(input_size))
        self.activation_func = activation_func
        self.activation_func_prime = activation_func_prime
        self.z = 0
        self.a = 0
        self.dw = 0

    def update_activation(self, x):
        self.z = x.dot(self.weights.T)
        self.a = self.activation_func(self.z)
        return self.a



# mnist.init()
x_train, y_train, x_test, y_test = mnist.load()


# Pre-process data
x_train, x_test = (x_train / 127.5) - 1, (x_test / 127.5) - 1
x_train = bias_trick(x_train)
x_test = bias_trick(x_test)
y_train, y_test = onehot_encode(y_train), onehot_encode(y_test)

x_train, y_train, x_val, y_val = train_val_split(x_train, y_train, 0.1)

layers = []
layers.append(Layer(785,32,sigmoid, sigmoid_prime))
layers.append(Layer(32,32,sigmoid, sigmoid_prime))
layers.append(Layer(32,10,softmax, None))

# Hyperparameters
batch_size = 128
learning_rate = 1
num_batches = x_train.shape[0] // batch_size
should_gradient_check = False
check_step = num_batches // 10
max_epochs = 20

# Tracking variables
TRAIN_LOSS = []
TEST_LOSS = []
VAL_LOSS = []
TRAIN_ACC = []
TEST_ACC = []
VAL_ACC = []


def train_loop():
    global x_train, y_train, x_val, y_val, x_test, y_test, layers
    for e in range(max_epochs):  # Epochs
        x_train, Y_train = shuffle_training_sets(x_train, y_train)
        for i in tqdm.trange(num_batches):
            X_batch = x_train[i * batch_size:(i+1) * batch_size]
            Y_batch = Y_train[i*batch_size:(i+1)*batch_size]

            layers = backpropagate(X_batch, Y_batch, layers, learning_rate)
            if i % check_step == 0:
                # Loss
                TRAIN_LOSS.append(cross_entropy_loss(x_train, Y_train, layers))
                TEST_LOSS.append(cross_entropy_loss(x_test, y_test, layers))
                VAL_LOSS.append(cross_entropy_loss(x_val, y_val, layers))

                TRAIN_ACC.append(calculate_accuracy(x_train, Y_train, layers))
                VAL_ACC.append(calculate_accuracy(x_val, x_val, layers))
                TEST_ACC.append(calculate_accuracy(x_test, y_test, layers))
                if should_early_stop(VAL_LOSS):
                    print(VAL_LOSS[-4:])
                    print("early stopping.")
                    return layers
    return layers


layers = train_loop()
plt.plot(TRAIN_LOSS, label="Training loss")
plt.plot(TEST_LOSS, label="Testing loss")
plt.plot(VAL_LOSS, label="Validation loss")
plt.legend()
plt.ylim([0, 0.15])
plt.show()

plt.clf()
plt.plot(TRAIN_ACC, label="Training accuracy")
plt.plot(TEST_ACC, label="Testing accuracy")
plt.plot(VAL_ACC, label="Validation accuracy")
plt.ylim([0.5, 1.0])
plt.legend()
plt.show()

plt.clf()
'''
w2 = w2[:, :-1]  # Remove bias
w2 = w2.reshape(10, 28, 28)
w = np.concatenate(w2, axis=0)
plt.imshow(w, cmap="gray")
plt.show()
'''

