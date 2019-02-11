import numpy as np
import matplotlib.pyplot as plt
import mnist
import tqdm


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


def train_val_split(X, Y, val_percentage):
    """
    Selects samples from the dataset randomly to be in the validation set.
    Also, shuffles the train set.
    --
    X: [N, num_features] numpy vector,
    Y: [N, 1] numpy vector
    val_percentage: amount of data to put in validation set
    """
    dataset_size = X.shape[0]
    idx = np.arange(0, dataset_size)
    np.random.shuffle(idx)

    train_size = int(dataset_size*(1-val_percentage))
    idx_train = idx[:train_size]
    idx_val = idx[train_size:]
    X_train, Y_train = X[idx_train], Y[idx_train]
    X_val, Y_val = X[idx_val], Y[idx_val]
    return X_train, Y_train, X_val, Y_val


def onehot_encode(Y, n_classes=10):
    onehot = np.zeros((Y.shape[0], n_classes))
    onehot[np.arange(0, Y.shape[0]), Y] = 1
    return onehot


def bias_trick(X):
    """
    X: shape[batch_size, num_features(784)]
    -- 
    Returns [batch_size, num_features+1 ]
    """
    return np.concatenate((X, np.ones((len(X), 1))), axis=1)


def check_gradient(X, targets, w, epsilon, computed_gradient):
    """
    Computes the numerical approximation for the gradient of w,
    w.r.t. the input X and target vector targets.
    Asserts that the computed_gradient from backpropagation is 
    correct w.r.t. the numerical approximation.
    --
    X: shape: [batch_size, num_features(784+1)]. Input batch of images
    targets: shape: [batch_size, num_classes]. Targets/label of images
    w: shape: [num_classes, num_features]. Weight from input->output
    epsilon: Epsilon for numerical approximation (See assignment)
    computed_gradient: Gradient computed from backpropagation. Same shape as w.
    """
    print("Checking gradient...")
    dw = np.zeros_like(w)
    for k in range(w.shape[0]):
        for j in range(w.shape[1]):
            new_weight1, new_weight2 = np.copy(w), np.copy(w)
            new_weight1[k,j] += epsilon
            new_weight2[k,j] -= epsilon
            loss1 = cross_entropy_loss(X, targets, new_weight1)
            loss2 = cross_entropy_loss(X, targets, new_weight2)
            dw[k,j] = (loss1 - loss2) / (2*epsilon)
    maximum_abosulte_difference = abs(computed_gradient-dw).max()
    assert maximum_abosulte_difference <= epsilon**2, "Absolute error was: {}".format(maximum_abosulte_difference)


def softmax(a):
    """
    Applies the softmax activation function for the vector a.
    --
    a: shape: [batch_size, num_classes]. Activation of the output layer before activation
    --
    Returns: [batch_size, num_classes] numpy vector
    """
    a_exp = np.exp(a)
    return a_exp / a_exp.sum(axis=1, keepdims=True)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def forward(X, w1, w2):
    """
    Performs a forward pass through the network
    --
    X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
    w: shape: [num_classes, num_features] numpy vector. Weight from input->output
    --
    Returns: [batch_size, num_classes] numpy vector
    """
    a1 = sigmoid(X.dot(w1.T))
    a2 = softmax(a1.dot(w2.T))

    return a2


def calculate_accuracy(X, targets, w1, w2):
    """
    Calculated the accuracy of the network.
    ---
    X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
    targets: shape: [batch_size, num_classes] numpy vector. Targets/label of images
    w: shape: [num_classes, num_features] numpy vector. Weight from input->output
    --
    Returns float
    """
    output = forward(X, w1, w2)
    predictions = output.argmax(axis=1)
    targets = targets.argmax(axis=1)
    return (predictions == targets).mean()


def cross_entropy_loss(X, targets, w1, w2):
    """
    Computes the cross entropy loss given the input vector X and the target vector.
    ---
    X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
    targets: shape: [batch_size, num_classes] numpy vector. Targets/label of images
    w: shape: [num_classes, num_features] numpy vector. Weight from input->output
    --
    Returns float
    """
    output = forward(X, w1, w2)
    assert output.shape == targets.shape 
    log_y = np.log(output)
    cross_entropy = -targets * log_y
    return cross_entropy.mean()


def gradient_descent(X, targets, w1, w2, learning_rate, should_check_gradient):
    """
    Performs gradient descents for all weights in the network.
    ---
    X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
    targets: shape: [batch_size, num_classes] numpy vector. Targets/label of images
    w: shape: [num_classes, num_features] numpy vector. Weight from input->output
    --
    Returns updated w, with same shape
    """

    # Since we are taking the .mean() of our loss, we get the normalization factor to be 1/(N*C)
    # If you take loss.sum(), the normalization factor is 1.
    # The normalization factor is identical for all weights in the network (For multi-layer neural-networks as well.)
    normalization_factor = X.shape[0] * targets.shape[1] # batch_size * num_classes
    outputs = forward(X, w1, w2)
    delta_k = - (targets - outputs)
    a = sigmoid(X.dot(w1.T))

    inside_parentheses = np.multiply(np.multiply(a, 1-a), np.dot(delta_k, w2))

    dw1 = np.dot(inside_parentheses.T, X)
    dw2 = delta_k.T.dot(a)

    dw1 = dw1 / normalization_factor  # Normalize gradient equally as we do with the loss
    dw2 = dw2 / normalization_factor  # Normalize gradient equally as we do with the loss
    assert dw1.shape == w1.shape, "dw shape was: {}. Expected: {}".format(dw1.shape, w1.shape)
    assert dw2.shape == w2.shape, "dw shape was: {}. Expected: {}".format(dw2.shape, w2.shape)

    if should_check_gradient:
        check_gradient(X, targets, w1, 1e-2,  dw1)
        check_gradient(X, targets, w2, 1e-2,  dw2)

    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2
    return w1, w2


#mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()

# Pre-process data
X_train, X_test = (X_train/127.5)-1, (X_test/127.5)-1
X_train = bias_trick(X_train)
X_test = bias_trick(X_test)
Y_train, Y_test = onehot_encode(Y_train), onehot_encode(Y_test)

X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, 0.1)


# Hyperparameters
batch_size = 128
learning_rate = 1
num_batches = X_train.shape[0] // batch_size
should_gradient_check = False
check_step = num_batches // 10
max_epochs = 40
layer_sizes = [785, 64, 10]

# Tracking variables
TRAIN_LOSS = []
TEST_LOSS = []
VAL_LOSS = []
TRAIN_ACC = []
TEST_ACC = []
VAL_ACC = []


def train_loop():
    w1 = np.random.uniform(-1, 1, (64, X_train.shape[1]))
    w2 = np.random.uniform(-1, 1, (Y_train.shape[1], 64))
    for e in range(max_epochs):  # Epochs
        for i in tqdm.trange(num_batches):
            X_batch = X_train[i*batch_size:(i+1)*batch_size]
            Y_batch = Y_train[i*batch_size:(i+1)*batch_size]

            w1, w2 = gradient_descent(X_batch,
                                 Y_batch,
                                 w1, w2,
                                 learning_rate,
                                 should_gradient_check)
            if i % check_step == 0:
                # Loss
                TRAIN_LOSS.append(cross_entropy_loss(X_train, Y_train, w1, w2))
                TEST_LOSS.append(cross_entropy_loss(X_test, Y_test, w1, w2))
                VAL_LOSS.append(cross_entropy_loss(X_val, Y_val, w1, w2))

                TRAIN_ACC.append(calculate_accuracy(X_train, Y_train, w1, w2))
                VAL_ACC.append(calculate_accuracy(X_val, Y_val, w1, w2))
                TEST_ACC.append(calculate_accuracy(X_test, Y_test, w1, w2))
                if should_early_stop(VAL_LOSS):
                    print(VAL_LOSS[-4:])
                    print("early stopping.")
                    return w1, w2
    return w1, w2


w1, w2 = train_loop()
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





