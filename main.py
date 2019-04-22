import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pickle
import os
from pprint import pprint
from matplotlib import pyplot as plt


def softplus(a):
    return np.log(1 + np.exp(a))


def tanh(a):
    term1 = np.exp(a) - np.exp(-a)
    term2 = np.exp(a) + np.exp(-a)
    return term1 / term2


def cos(a):
    return np.cos(a)


def derivative_softplus(a):
    pass


def derivative_tanh(a):
    pass


def derivative_cos(a):
    pass


def activation_function(a):
    return non_linearity(a)


def derivative_activation_function(a):
    pass


def get_hidden_layer_representation(input, weights_1):
    # TODO add the bias (1)
    # input : batch_size x D , weights_1 : M x D

    a = input.dot(weights_1.T)

    a = activation_function(a)

    # TODO add the bias to a. Dimensions right now: batch_size x M
    return a


def forward(input, weights_1, weights_2):
    # TODO add BIAS
    # pass the input through the hidden layer
    hidden_layer_representation = get_hidden_layer_representation(input, weights_1)

    # calculate dot product between weights_2 and hidden layer representation
    # hidden layer_representation : B x M
    # weights_2 : M x K
    output = hidden_layer_representation.dot(weights_2.T)

    # output now has dimensions: batch_size x K
    output = softmax(output)

    return output, hidden_layer_representation, weights_1, weights_2


def backward(labels, output, hidden_layer_rep, weights_1, weights_2, lamda):

    # gradients for the weights between the hidden layer and the softmax layer
    grad_weights_2 = (labels - output).T.dot(hidden_layer_rep) - lamda * weights_2
    

# not vectorized...it will be used for comparison with the vectorized version
def get_loss(batch_size, classes, labels, output, lamda, weights):
    # first calculate the norm
    regularization = (lamda / 2) * (np.power(np.linalg.norm(weights), 2))
    sum = 0
    for n in range(0, batch_size):
        for k in range(0, classes):
            sum += labels[n, k] * np.log(output[n, k])

    return sum - regularization


# vectorized version of the loss function
def get_loss_vectorized(labels, output, lamda, weights):
    regularization = (lamda / 2) * (np.power(np.linalg.norm(weights), 2))
    output = np.log(output)
    output = labels * output
    output = np.sum(output, 1)
    output = np.sum(output, 0)
    loss = output - regularization
    return loss


def softmax(a):
    # this version of softmax is more numerical stable
    s = np.max(a, axis=1)
    s = np.expand_dims(s, 1)  # unsqueeze
    e_x = np.exp(a - s)
    div = np.sum(e_x, axis=1)
    div = np.expand_dims(div, 1)  # unsqueeze
    return e_x / div


def batch_yielder(batch_size=128):
    pass


def train():
    pass


def test():
    pass


# a function to load the mnist data
def load_mnist_data(path):
    train_files = ['train{}.txt'.format(i) for i in range(0, 10)]
    test_files = ['test{}.txt'.format(i) for i in range(0, 10)]
    print('Total train files: {}'.format(len(train_files)) + '\n' + 'Total test files: {}'.format(len(test_files)))

    data = []
    for idx, file in enumerate(train_files):
        with open(os.path.join(path, file), 'r') as train_file:
            data += train_file.readlines()
    train_data = np.array(np.array([[j for j in i.split(" ")] for i in data], dtype='int') / 255)
    print('Total training images with shape: ', train_data.shape)

    data = []
    for idx, file in enumerate(test_files):
        with open(os.path.join(path, file), 'r') as test_file:
            data += test_file.readlines()
    test_data = np.array(np.array([[j for j in i.split(" ")] for i in data], dtype='int') / 255)
    print('Total testing images with shape: ', test_data.shape)

    labels_data = []
    for idx, file in enumerate(train_files):
        with open(os.path.join(path, file), 'r') as train_file:
            for _ in train_file:
                labels_data.append([1 if i == idx else 0 for i in range(0, 10)])

    train_labels = np.array(labels_data, dtype='int')
    print('Total training labels: ', train_labels.shape)

    labels_data = []
    for idx, file in enumerate(test_files):
        with open(os.path.join(path, file), 'r') as test_file:
            for _ in test_file:
                labels_data.append([1 if i == idx else 0 for i in range(0, 10)])

    test_labels = np.array(labels_data, dtype='int')
    print('Total testing labels: ', test_labels.shape)

    return train_data, train_labels, test_data, test_labels


# a function to load the cifar data
def load_cifar_10_data():
    pass


# load_mnist_data('/home/sotiris/PycharmProjects/mnist_data/')
non_linearity = cos
# input = np.array([[1, 2, 3, 6],
#                   [2, 4, 5, 6],
#                   [1, 2, 3, 6]])
# print(input)
# print(input.shape)
# input = np.array([13, 46, 79])
# print(softmax(input))
# activation_function(1.2)
# labels = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
# output = np.array([[0.05, 0.05, 0.05, 0.8, 0.05], [0.05, 0.05, 0.05, 0.8, 0.05]])
# weights1 = np.array([[1, 2, 3, 4, 5],
#                      [1, 2, 3, 4, 5]])
#
# weights2 = np.array([[6, 7, 8, 9, 10],
#                      [1, 2, 3, 4, 5]])
# weights = np.concatenate((weights1, weights2))
# print(weights.flatten())
# print(get_loss(1, 5, labels, output))
# print(get_loss_vectorized(labels, output, 0.1, weights))

