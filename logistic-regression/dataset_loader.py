import numpy as np
import h5py

def print_infos(x_train, y_train, x_test, y_test):
    shape_px = x_test.shape[0]
    print ("Number of training examples: m_train = " + str(x_train.shape[0]))
    print ("Number of testing examples: m_test = " + str(x_test.shape[0]))
    print ("Height/Width of each image: num_px = " + str(shape_px))
    print ("Each image is of size: (", str(shape_px), ", " , str(shape_px) ,", 3)")
    print ("x_train shape: ", str(x_train.shape))
    print ("y_train shape: ", str(y_train.shape))
    print ("x_test shape: ",str(x_test.shape))
    print ("y_test shape: ", str(y_test.shape))

def load_dataset():

    train_dataset = h5py.File('../datasets/train_catvnoncat.h5', "r")
    x_train = np.array(train_dataset["train_set_x"][:])
    y_train = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('../datasets/test_catvnoncat.h5', "r")
    x_test = np.array(test_dataset["test_set_x"][:])
    y_test = np.array(test_dataset["test_set_y"][:])

    y_train = y_train.reshape((1, y_train.shape[0]))
    y_test = y_test.reshape((1, y_test.shape[0]))

    print_infos(x_train, y_train, x_test, y_test)

    return x_train, y_train, x_test, y_test

def flatten_dataset(x_train, x_test):

    x_train_flatten = x_train.reshape(x_train.shape[0], -1).T
    x_test_flatten = x_test.reshape(x_test.shape[0], -1).T

    print ("x_train_flatten shape: " + str(x_train_flatten.shape))
    print ("x_test_flatten shape: " + str(x_test_flatten.shape))

    x_train = x_train_flatten/255.
    x_test = x_test_flatten/255.

    return x_train, x_test
