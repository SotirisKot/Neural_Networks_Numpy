import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pickle
import os
from pprint import pprint
from matplotlib import pyplot as plt


def activation_function(a, h):
    # the first function is a soft implementation of relu..named softplus
    if h == 1:
        return np.log(1 + np.exp(a))
    # the second activation function is tanh
    elif h == 2:
        term1 = np.exp(a) - np.exp(-a)
        term2 = np.exp(a) + np.exp(-a)
        return term1 / term2
    else:
        # the third activation function is a cos.
        return np.cos(a)


def forward():
    pass


def backward():
    pass


# not vectorized...it will be used for comparison with the vectorized version
def loss(N, K, t, y, lamb, weights):
    # first calculate the norm
    regularization = (-1.0 * lamb / 2) * (np.power(np.linalg.norm(weights), 2))
    sum = 0
    for n in range(1, N+1):
        for k in range(1, K+1):
            sum += t[n, k]*np.log(y[n,k])


def softmax():
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


load_mnist_data('/home/sotiris/PycharmProjects/mnist_data/')
