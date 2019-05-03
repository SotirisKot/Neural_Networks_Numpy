import re
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import _pickle as cPickle
import random
import os
from pprint import pprint
from matplotlib import pyplot as plt

random.seed(1997)


###### ACTIVATION FUNCTIONS ######

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
    return 1 / (1 + np.exp(- a))


def derivative_tanh(a):
    return 1 - np.power(np.tanh(a), 2)


def derivative_cos(a):
    return -(np.sin(a))


def activation_function(a):
    return non_linearity(a)


def derivative_activation_function(a):
    return derivative_function(a)

###################################


# calculates the result of the one hidden layer and adds the bias to the result
def get_hidden_layer_representation(input, weights_1):
    # input : batch_size x (D + 1) , weights_1 : M x (D + 1)
    a = input.dot(weights_1.T)

    a = activation_function(a)

    # add the bias to a. Dimensions right now: batch_size x M
    # adding the bias the new dimensions will be : batch_size x (M + 1)
    a_final = np.ones((a.shape[0], a.shape[1] + 1), dtype=float)
    a_final[:, 1:] = a

    return a_final


# the forward pass of our system...it calculates the loss and the softmax probabilities for the K categories
def forward(input, labels, weights_1, weights_2):
    # pass the input through the hidden layer
    hidden_layer_representation = get_hidden_layer_representation(input, weights_1)

    # calculate dot product between weights_2 and hidden layer representation
    # hidden layer_representation : batch_size x (M + 1)
    # weights_2 : K x (M + 1)
    output = hidden_layer_representation.dot(weights_2.T)

    # output now has dimensions: batch_size x K
    output = softmax(output)

    # calculate the loss
    # first flatten the parameters
    # concatenate the two weight matrices
    # weights = np.concatenate((weights_1.flatten(), weights_2.flatten()))
    loss = get_loss_vectorized(labels, output, weights_1, weights_2)

    return loss, output, hidden_layer_representation


# this function is used during testing...it does a forward pass without calculating the loss..it returns the softmax
# probabilities.
def inference(input, weights_1, weights_2):
    # pass the input through the hidden layer
    hidden_layer_representation = get_hidden_layer_representation(input, weights_1)

    # calculate dot product between weights_2 and hidden layer representation
    # hidden layer_representation : B x (M + 1)
    # weights_2 : K x (M + 1)
    output = hidden_layer_representation.dot(weights_2.T)

    # output now has dimensions: batch_size x K
    output = softmax(output)

    return output


# this is the backward pass that calculates the gradients that are used in stochastic gradient ascent
def backward(labels, output, hidden_layer_rep, weights_1, weights_2, input):
    # gradients for the weights between the hidden layer and the softmax layer
    grad_weights_2 = (labels - output).T.dot(hidden_layer_rep) - lamda * weights_2

    # remove the bias from weights_2
    weights_2_temp = np.copy(weights_2[:, 1:])

    derivative_result = derivative_activation_function(input.dot(weights_1.T))

    grad_weights_1_tmp = (labels - output).dot(weights_2_temp) * derivative_result

    grad_weights_1 = grad_weights_1_tmp.T.dot(input) - lamda * weights_1

    return grad_weights_1, grad_weights_2


##### LOSS FUNCTIONS ######

# not vectorized...it will be used for comparison with the vectorized version
def get_loss(labels, output, weights):
    # first calculate the norm
    regularization = (0.5 * lamda) * (np.power(np.linalg.norm(weights), 2))
    sum = 0
    for n in range(0, batch_size):
        for k in range(0, K):
            sum += labels[n, k] * np.log(output[n, k])

    return sum - regularization


# vectorized version of the loss function...performs must faster.
def get_loss_vectorized(labels, output, weights1, weights2):
    # these two calculations of the regularization term are the same
    # regularization = (lamda / 2) * (np.power(np.linalg.norm(weights), 2))
    regularization = (lamda / 2) * (np.sum(np.square(weights1)) + np.sum(np.square(weights2)))
    output = np.log(output)
    output = labels * output
    output = np.sum(output, 1)
    output = np.sum(output, 0)
    loss = output - regularization
    return loss

###############################


# softmax function...turns the output to probilities that sum to 1..tends to favor large numbers.
def softmax(x, ax=1):
    m = np.max(x, axis=ax, keepdims=True)  # max per row
    p = np.exp(x - m)
    return p / np.sum(p, axis=ax, keepdims=True)


# a function that yields a batch from the dataset.
def batch_yielder(data, data_labels):
    for idx in range(0, data.shape[0], batch_size):
        input_b = data[idx:idx + batch_size]
        labels_b = data_labels[idx:idx + batch_size]
        yield input_b, labels_b


# the function that trains the model. Initializes the weights based on the normal distribution
# runs for #epochs, does a forward and a backward pass for each batch updating the weights.
def train():
    ####################################
    # worked best
    # weights_1 ----> normal distribution with a mean of 0 and standard deviation of 1.
    # weights_2 ----> zeros
    ####################################

    # initialize the weights
    # for weights_1 we will use xavier initialization or normal distribution
    # weights_2 will be initialized to zeros or xavier or normal distribution
    # weights_1 = np.random.rand(M, D + 1) * np.sqrt(1 / (D + 1 + M))
    # weights_1[:, 0] = 1.0
    #
    # # weights_2 = np.zeros((K, M+1))
    # weights_2 = np.random.rand(K, M + 1) * np.sqrt(1 / (M + 1 + K))
    # weights_2[:, 0] = 1.0

    center = 0
    s = np.sqrt(1 / (D + 1))

    # Initialize the weights
    weights_2 = np.zeros((K, M + 1))

    # We use this in order for our activation function to be more effective
    weights_1 = np.random.normal(center, s, (M, D + 1))

    for _ in tqdm(range(epochs)):
        epoch_loss = []
        iterator = batch_yielder(train_data, train_labels)
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


# a function to test a batch...if does a forward pass for a batch and returns the predictions for it.
def test_batch():
    # we do forward passes with the learned weights and we predict
    iterator = tqdm(batch_yielder(test_data, test_labels))
    true = 0
    total = 0
    for batch in tqdm(iterator):
        input_data = batch[0]
        labels = batch[1]

        ### forward
        output = inference(input_data, learned_weights_1, learned_weights_2)
        predictions = np.argmax(output, 1)
        labels = np.argmax(labels, 1)

        for idx in range(labels.shape[0]):
            if predictions[idx] == labels[idx]:
                true += 1
            total += 1

    print(true)
    print(total)
    print("Accuracy : {}/{} ... {}".format(true, total, float(true / total)))


# the test function that was used..it tests the entire test set at once and logs some information.
def test():
    output = inference(test_data, learned_weights_1, learned_weights_2)
    predictions = np.argmax(output, 1)
    accuracy = np.mean(predictions == np.argmax(test_labels, 1))
    print('Accuracy : {}'.format(accuracy))
    print("Writing to file: ")
    with open('machine_learning_results.txt', 'a') as file:
        file.write("DATASET: " + " " + dataset)
        file.write('\n')
        file.write("HIDDEN_SIZE " + " " + str(M))
        file.write('\n')
        file.write("Learning Rate: " + ' ' + str(lr))
        file.write('\n')
        file.write("Lamda: " + ' ' + str(lamda))
        file.write('\n')
        file.write("Accuracy: " + ' ' + str(accuracy))
        file.write('\n')
        file.write("Activation Function: " + ' ' + str(non_linearity))
        file.write('\n')
        file.write("######################################################################")
        file.write('\n')


# a function to show an image of the MNIST dataset
def show_image_mnist(image_data):
    # plot 5 random images from the training set
    n = 100
    sqrt_n = int(n ** 0.5)
    samples = np.random.randint(train_data_old.shape[0], size=n)

    plt.figure(figsize=(11, 11))

    cnt = 0
    for i in samples:
        cnt += 1
        plt.subplot(sqrt_n, sqrt_n, cnt)
        plt.subplot(sqrt_n, sqrt_n, cnt).axis('off')
        plt.imshow(train_data_old[i].reshape(28, 28), cmap='gray')

    plt.show()


# a function to show an image of the CIFAR dataset
def plot_cifar(ind):
    arr = train_data_old[ind]
    R = arr[0:1024].reshape(32, 32)
    G = arr[1024:2048].reshape(32, 32)
    B = arr[2048:].reshape(32, 32)

    img = np.dstack((R, G, B))
    title = re.sub('[!@#$b]', '', str(labels_names[np.argmax(train_labels[ind])]))
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    ax.imshow(img, interpolation='bicubic')
    ax.set_title('Category = ' + title, fontsize=15)
    plt.show()


# this function performs the gradient check..For an epsilon of 1e-6 the difference must be smaller than that.
# for the mnist dataset the difference for weights_1 ---> 1e-7 and for weights_2 ---> 1e-10
def gradient_check():
    epsilon = 1e-6

    # create two random weight matrices
    weights_1_tmp = np.random.rand(M, D + 1) * np.sqrt(1 / (D + 1 + M))
    weights_2_tmp = np.random.rand(K, M + 1) * np.sqrt(1 / (M + 1 + K))

    weights_1_tmp[:, 0] = 1.0
    weights_2_tmp[:, 0] = 1.0

    # create a fake train batch (of size 8)
    b_size = 4
    fake_input = train_data[:b_size]
    fake_labels = train_labels[:b_size]

    # calculate gradients with backpropagation
    loss, output, hidden_layer_output = forward(fake_input, fake_labels, weights_1_tmp, weights_2_tmp)
    grad_weights_1, grad_weights_2 = backward(fake_labels,
                                              output,
                                              hidden_layer_output,
                                              weights_1_tmp,
                                              weights_2_tmp,
                                              fake_input)

    # calculate gradients with two-sided epsilon method
    grad_check_for_w1 = np.zeros((M, D + 1))
    for i in tqdm(range(grad_check_for_w1.shape[0])):
        for j in range(grad_check_for_w1.shape[1]):
            w1 = np.copy(weights_1_tmp)
            w1[i, j] += epsilon
            e1, _, _ = forward(fake_input, fake_labels, w1, weights_2_tmp)

            w1 = np.copy(weights_1_tmp)
            w1[i, j] -= epsilon
            e2, _, _ = forward(fake_input, fake_labels, w1, weights_2_tmp)

            grad_check_for_w1[i, j] = (e1 - e2) / (2 * epsilon)

    # compute the difference for w1
    # it is the Euclidean distance normalized by the sum of the norms
    numerator = np.linalg.norm(grad_weights_1 - grad_check_for_w1)
    denominator = np.linalg.norm(grad_check_for_w1) + np.linalg.norm(grad_weights_1)
    difference = numerator / denominator
    print('The difference for weights_1 is: {}'.format(difference))

    grad_check_for_w2 = np.zeros((K, M + 1))
    for i in tqdm(range(grad_check_for_w2.shape[0])):
        for j in range(grad_check_for_w2.shape[1]):
            w2 = np.copy(weights_2_tmp)
            w2[i, j] += epsilon
            e1, _, _ = forward(fake_input, fake_labels, weights_1_tmp, w2)

            w2 = np.copy(weights_2_tmp)
            w2[i, j] -= epsilon
            e2, _, _ = forward(fake_input, fake_labels, weights_1_tmp, w2)

            grad_check_for_w2[i, j] = (e1 - e2) / (2 * epsilon)

    # compute the difference for w2
    # it is the Euclidean distance normalized by the sum of the norms
    numerator = np.linalg.norm(grad_weights_2 - grad_check_for_w2)
    denominator = np.linalg.norm(grad_check_for_w2) + np.linalg.norm(grad_weights_2)
    difference = numerator / denominator
    print('The difference for weights_2 is: {}'.format(difference))


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
    train_batches = ['data_batch_{}'.format(i) for i in range(1, 6)]
    data = []
    train_labels_tmp = np.zeros((50000, 10))

    batch_dicts = []
    for batch_name in train_batches:
        batch_dicts.append(load_cifar_10_batch(batch_name))

    img_idx = 0
    for batch_dict in batch_dicts:
        for img_data in batch_dict['data']:
            data.append(img_data)

        for img_label in batch_dict['labels']:
            train_labels_tmp[img_idx][img_label] = 1
            img_idx += 1

    train_data_tmp = np.asarray(data)
    train_data_tmp = train_data_tmp / 255

    test_batch_dict = load_cifar_10_batch('test_batch', True)
    data = []
    test_labels_tmp = np.zeros((10000, 10))
    for img in test_batch_dict['data']:
        data.append(img)

    img_idx = 0
    for img_label in test_batch_dict['labels']:
        test_labels_tmp[img_idx][img_label] = 1
        img_idx += 1

    test_data_tmp = np.asarray(data)
    test_data_tmp = test_data_tmp / 255

    print('Total training images with shape: ', train_data_tmp.shape)
    print('Total testing images with shape: ', test_data_tmp.shape)
    print('Total training labels: ', train_labels_tmp.shape)
    print('Total test labels: ', test_labels_tmp.shape)

    return train_data_tmp, train_labels_tmp, test_data_tmp, test_labels_tmp


# unpickle a batch from the cifar_10 dataset
def load_cifar_10_batch(batch_name, test_bool=False):
    data_batch = cPickle.load(open('/home/sotiris/PycharmProjects/cifar-10-batches-py/{}'.format(batch_name), 'rb'),
                              encoding='latin1')

    return data_batch


# if set to True an image will be plotted
plot_an_image = False

# if set to True gradient check will be performed
do_gradient_check = False


###### CODE FOR EXPERIMENTS ######

# This part of the code was used for testing different parameters
# batch_size = 100
# epochs = 200
# K = 10
# datasets = ['CIFAR', 'MNIST']
# hidden_sizes = [100, 200, 300]
# activation_functions = [softplus, tanh, cos]
# lr_s = [0.01, 0.001]
# lamdas = [0.1, 0.05, 0.01]
#
# for dataset in datasets:
#     if dataset == 'MNIST':
#         path = '/home/sotiris/PycharmProjects/mnist_data/'
#         train_data_old, train_labels, test_data_old, test_labels = load_mnist_data()
#         if plot_an_image:
#             show_image_mnist(train_data_old)
#     else:
#         path = '/home/sotiris/PycharmProjects/cifar_data/'
#         labels_dict = cPickle.load(
#             open('/home/sotiris/PycharmProjects/cifar-10-batches-py/{}'.format('batches.meta'), 'rb'))
#         labels_names = labels_dict['label_names']
#         train_data_old, train_labels, test_data_old, test_labels = load_cifar_10_data()
#
#         # plot a random sample from the train set
#         if plot_an_image:
#             ind = np.random.randint(0, 50000)
#             plot_cifar(ind)
#
#     train_data = np.ones((train_data_old.shape[0], train_data_old.shape[1] + 1), dtype=float)
#     train_data[:, 1:] = train_data_old
#
#     test_data = np.ones((test_data_old.shape[0], test_data_old.shape[1] + 1), dtype=float)
#     test_data[:, 1:] = test_data_old
#
#     idx_list = [i for i in range(train_labels.shape[0])]
#     shuffle(idx_list)
#     train_data = train_data[idx_list]
#     train_labels = train_labels[idx_list]
#     total_batches = train_data.shape[0] // batch_size
#
#     D = train_data_old.shape[1]
#     print(D)
#     # we must also shuffle the test data
#     idx_list = [i for i in range(test_labels.shape[0])]
#     shuffle(idx_list)
#     test_data = test_data[idx_list]
#     test_labels = test_labels[idx_list]
#
#     for non_linearity in tqdm(activation_functions):
#
#         if non_linearity == tanh:
#             derivative_function = derivative_tanh
#         elif non_linearity == cos:
#             derivative_function = derivative_cos
#         else:
#             derivative_function = derivative_softplus
#
#         for M in hidden_sizes:
#             for lr in lr_s:
#                 for lamda in lamdas:
#                     learned_weights_1, learned_weights_2 = train()
#                     test()

############## some test code #####################
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

#################################


###### DATASET ######

# the two names are 'MNIST' and 'CIFAR'
dataset = 'CIFAR'

if dataset == 'MNIST':
    path = '/home/sotiris/PycharmProjects/mnist_data/'
    train_data_old, train_labels, test_data_old, test_labels = load_mnist_data()
    if plot_an_image:
        show_image_mnist(train_data_old)
else:
    path = '/home/sotiris/PycharmProjects/cifar_data/'
    labels_dict = cPickle.load(open('/home/sotiris/PycharmProjects/cifar-10-batches-py/{}'.format('batches.meta'), 'rb'))
    labels_names = labels_dict['label_names']
    train_data_old, train_labels, test_data_old, test_labels = load_cifar_10_data()

    # plot a random sample from the train set
    if plot_an_image:
        ind = np.random.randint(0, 50000)
        plot_cifar(ind)

#####################


###### MODEL PARAMETERS ######

batch_size = 100
epochs = 300
lr = 0.001
lamda = 0.1
M = 300
D = train_data_old.shape[1]
print(D)
K = 10

non_linearity = tanh
derivative_function = derivative_tanh

##############################


###### CREATE SPLITS ######
# also add the bias.
train_data = np.ones((train_data_old.shape[0], train_data_old.shape[1] + 1), dtype=float)
train_data[:, 1:] = train_data_old

test_data = np.ones((test_data_old.shape[0], test_data_old.shape[1] + 1), dtype=float)
test_data[:, 1:] = test_data_old

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

if do_gradient_check:
    print('Performing gradient check...might take some time...')
    gradient_check()

learned_weights_1, learned_weights_2 = train()
###################

###### EVALUATE ######
test()
######################
