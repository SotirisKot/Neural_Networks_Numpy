from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pickle
import random
import os
from pprint import pprint
from matplotlib import pyplot as plt
random.seed(1997)


def softplus(a):
    # return np.log(1 + np.exp(a)) this version is not stable for large a...overflows
    return np.log(1 + np.exp(- np.abs(a))) + np.maximum(a, 0)  # numerical stable


def tanh(a):
    # term1 = np.exp(a) - np.exp(-a)
    # term2 = np.exp(a) + np.exp(-a)
    # return term1 / term2
    return np.tanh(a)  # numerical stable


def cos(a):
    return np.cos(a)


def derivative_softplus(a):
    # TODO try the other derivative as well
    return 1 / (1 + np.exp(- a))


def derivative_tanh(a):
    return 1 - np.power(np.tanh(a), 2)


def derivative_cos(a):
    return -(np.sin(a))


def activation_function(a):
    return non_linearity(a)


def derivative_activation_function(a):
    return derivative_function(a)


def get_hidden_layer_representation(input, weights_1):
    # TODO add the bias (1)
    # input : batch_size x D , weights_1 : M x D

    a = input.dot(weights_1.T)

    a = activation_function(a)

    # TODO add the bias to a. Dimensions right now: batch_size x M
    return a


def forward(input, labels, weights_1, weights_2):
    # TODO add BIAS
    # pass the input through the hidden layer
    hidden_layer_representation = get_hidden_layer_representation(input, weights_1)

    # calculate dot product between weights_2 and hidden layer representation
    # hidden layer_representation : B x M
    # weights_2 : M x K
    output = hidden_layer_representation.dot(weights_2.T)

    # output now has dimensions: batch_size x K
    output = softmax(output)

    # calculate the loss
    # first flatten the parameters
    # concatenate the two weight matrices
    weights = np.concatenate((weights_1.flatten(), weights_2.flatten()))
    loss = get_loss_vectorized(labels, output, weights)

    return loss, output, hidden_layer_representation


def inference(input, weights_1, weights_2):
    # pass the input through the hidden layer
    hidden_layer_representation = get_hidden_layer_representation(input, weights_1)

    # calculate dot product between weights_2 and hidden layer representation
    # hidden layer_representation : B x M
    # weights_2 : M x K
    output = hidden_layer_representation.dot(weights_2.T)

    # output now has dimensions: batch_size x K
    output = softmax(output)

    return output


def backward(labels, output, hidden_layer_rep, weights_1, weights_2, input):

    # gradients for the weights between the hidden layer and the softmax layer
    grad_weights_2 = (labels - output).T.dot(hidden_layer_rep) - lamda * weights_2

    # TODO remove the bias from weights_2
    # weights_2_temp = np.copy(weights_2[:, 1:])

    derivative_result = derivative_activation_function(input.dot(weights_1.T))

    grad_weights_1_tmp = (labels - output).dot(weights_2)

    grad_weights_1_tmp = grad_weights_1_tmp * derivative_result

    grad_weights_1 = grad_weights_1_tmp.T.dot(input) - lamda * weights_1

    return grad_weights_1, grad_weights_2


# not vectorized...it will be used for comparison with the vectorized version
def get_loss(labels, output, weights):
    # first calculate the norm
    regularization = (lamda / 2) * (np.power(np.linalg.norm(weights), 2))
    sum = 0
    for n in range(0, batch_size):
        for k in range(0, K):
            sum += labels[n, k] * np.log(output[n, k])

    return sum - regularization


# vectorized version of the loss function
def get_loss_vectorized(labels, output, weights):
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


def batch_yielder(data, data_labels):
    for idx in range(0, data.shape[0], batch_size):
        input_b = data[idx:idx+batch_size]
        labels_b = data_labels[idx:idx+batch_size]
        yield input_b, labels_b


def train():
    # initialize the weights
    # for weights_1 we will use xavier initialization
    # weights_2 will be initialized to zeros or xavier
    weights_1 = np.random.rand(M, D)*np.sqrt(1/(D + M))

    weights_2 = np.zeros((K, M))
    # weights_2 = np.random.rand(K, M)*np.sqrt(1/(M + K))

    for epoch in range(epochs):
        epoch_loss = []
        iterator = tqdm(batch_yielder(train_data, train_labels))
        for batch in iterator:
            ### prepare data
            input_data = batch[0]
            labels = batch[1]

            ### forward
            loss, output, hidden_layer_output = forward(input_data, labels, weights_1, weights_2)
            epoch_loss.append(loss)

            ### backward
            grad_weights_1, grad_weights_2 = backward(labels,
                                                      output,
                                                      hidden_layer_output,
                                                      weights_1,
                                                      weights_2,
                                                      input_data)

            ### update the weights based on the learning rate
            weights_1 += lr * grad_weights_1
            weights_2 += lr * grad_weights_2

        print("Epoch average loss: ", sum(epoch_loss) / total_batches)
    # --------------------------------------
    print("Training ended...")
    return weights_1, weights_2


def test():
    # we do forward passes with the learned weights and we predict
    iterator = tqdm(batch_yielder(test_data, test_labels))
    for batch in tqdm(iterator):
        input_data = batch[0]
        labels = batch[1]

        ### forward
        output = inference(input_data, learned_weights_1, learned_weights_2)


def show_image(image_data):
    plt.imshow(image_data.reshape(28, 28), interpolation='nearest')
    plt.show()


# a function to load the mnist data
def load_mnist_data():
    train_files = ['train{}.txt'.format(i) for i in range(0, 10)]
    test_files = ['test{}.txt'.format(i) for i in range(0, 10)]
    print('Total train files: {}'.format(len(train_files)) + '\n' + 'Total test files: {}'.format(len(test_files)))

    data = []
    for idx, file in enumerate(train_files):
        with open(os.path.join(path, file), 'r') as train_file:
            data += train_file.readlines()
    train_data_tmp = np.array(np.array([[j for j in i.split(" ")] for i in data], dtype='int') / 255)
    print('Total training images with shape: ', train_data_tmp.shape)

    data = []
    for idx, file in enumerate(test_files):
        with open(os.path.join(path, file), 'r') as test_file:
            data += test_file.readlines()
    test_data_tmp = np.array(np.array([[j for j in i.split(" ")] for i in data], dtype='int') / 255)
    print('Total testing images with shape: ', test_data_tmp.shape)

    labels_data = []
    for idx, file in enumerate(train_files):
        with open(os.path.join(path, file), 'r') as train_file:
            for _ in train_file:
                labels_data.append([1 if i == idx else 0 for i in range(0, 10)])

    train_labels_tmp = np.array(labels_data, dtype='int')
    print('Total training labels: ', train_labels_tmp.shape)

    labels_data = []
    for idx, file in enumerate(test_files):
        with open(os.path.join(path, file), 'r') as test_file:
            for _ in test_file:
                labels_data.append([1 if i == idx else 0 for i in range(0, 10)])

    test_labels_tmp = np.array(labels_data, dtype='int')
    print('Total testing labels: ', test_labels_tmp.shape)

    return train_data_tmp, train_labels_tmp, test_data_tmp, test_labels_tmp


# a function to load the cifar data
def load_cifar_10_data():
    pass


###### MODEL PARAMETERS ######

batch_size = 120
epochs = 10
lr = 0.001
lamda = 0.01
M = 100
D = 784
K = 10
non_linearity = tanh
derivative_function = derivative_tanh

##############################


###### DATASET ######

dataset = 'MNIST'
if dataset == 'MNIST':
    path = '/home/sotiris/PycharmProjects/mnist_data/'
else:
    path = '/home/sotiris/PycharmProjects/cifar_data/'

#####################

###### CREATE SPLITS ######

train_data, train_labels, test_data, test_labels = load_mnist_data()

###########################

###### TRAIN ######

# first shuffle the train data
idx_list = [i for i in range(train_labels.shape[0])]
shuffle(idx_list)
train_data = train_data[idx_list]
train_labels = train_labels[idx_list]
total_batches = train_data.shape[0] // batch_size

# we must also shuffle the test data
idx_list = [i for i in range(test_labels.shape[0])]
shuffle(idx_list)
test_data = test_data[idx_list]
test_labels = test_labels[idx_list]

learned_weights_1, learned_weights_2 = train()
###################

###### EVALUATE ######
# test()
######################


# load_mnist_data('/home/sotiris/PycharmProjects/mnist_data/')
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
