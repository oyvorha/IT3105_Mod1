import mnist.mnist_basics as basics
import random
import tflowtools as tools
from GANN import *


"""
Functions for mnist preprocessing
"""


def choose_random_subset(w_img, w_lab):
    images = []
    labels = []
    used = []
    while len(images) < w_img.shape[0] * 0.1:
        random_int = random.randint(0, w_img.shape[0]-1)
        if random_int not in used:
            used.append(random_int)
            images.append(w_img[random_int])
            labels.append(w_lab[random_int])
    return np.asarray(images), np.asarray(labels)


def scale(data, d_max=255):
    return data/d_max


def labels_to_one_hot(labels):
    one_hot = []
    for label in labels:
        for number in label:
            one_hot.append(tools.int_to_one_hot(number, 10))
    return np.asarray(one_hot)


def cfunc_mnist():
    input_vec, target = choose_random_subset(basics.load_mnist()[0], basics.load_mnist()[1])
    input_vec = scale(input_vec)
    # Flatten cases such that image is ready for input into the NN, dim = (784,1)
    flat_cases = []
    for image in input_vec:
        flat_cases.append(basics.flatten_image(image))
    target = labels_to_one_hot(target)
    cases = []
    for i in range(len(flat_cases)):
        cases.append([flat_cases[i], target[i]])
    return cases


"""
Make NN with Gann for MNIST
"""


def NN_mnist(dims=[784, 200, 10], lrate=0.1, showint=2, mbs=10, vint=100, softmax=False):
    cman = Caseman(cfunc_mnist)
    gann_mnist = Gann(dims, cman, lrate, showint, mbs, vint, softmax)
    gann_mnist.run(100, None, False, 1)
