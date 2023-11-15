import numpy as np

# Load the dataset
X_train = np.load('train_images.npy')
y_train = np.load('train_labels.npy')
X_test = np.load('test_images.npy')
y_test = np.load('test_labels.npy')

# Preprocess the dataset
X_train = X_train / 255.0
X_test = X_test / 255.0
num_classes = 26
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

# Initialize the weights and biases
input_size = 784
hidden_size = 128
output_size = num_classes
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Define the forward propagation function
def forward(X):
    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0, Z1)
    Z2 = np.dot(A1, W2) + b2
    exp_scores = np.exp(Z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs

# Define the loss function
def loss(X, y):
    probs = forward(X)
    correct_logprobs = -np.log(probs[range(len(y)), y])
    data_loss = np.sum(correct_logprobs)
    return 1./len(X) * data_loss

# Define the backward propagation function
def backward(X, y, probs):
    delta3 = probs
    delta3[range(len(y)), y] -= 1
    dW2 = np.dot(A1.T, delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)
    delta2 = np.dot(delta3, W2.T)
    delta2[A1 <= 0] = 0
    dW1 = np.dot(X.T, delta2)
    db1 = np.sum(delta2, axis=0)
    return dW1, db1, dW2, db2

# Train the neural network
learning_rate = 0.1
for i in range(10000):
    probs = forward(X_train)
    cost = loss(X_train, y_train)
    dW1, db1, dW2, db2 = backward(X_train, y_train, probs)
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    if i % 1000 == 0:
        print("Iteration {}: loss = {}".format(i, cost))

# Evaluate the performance of the trained neural network
train_accuracy = np.mean(np.argmax(forward(X_train), axis=1) == np.argmax(y_train, axis=1))
test_accuracy = np.mean(np.argmax(forward(X_test), axis=1) == np.argmax(y_test, axis=1))
print("Train accuracy: {}".format(train_accuracy))
print("Test accuracy: {}".format(test_accuracy))
