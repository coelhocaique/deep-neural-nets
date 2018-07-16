import numpy as np
import numpy as np
import h5py

def load_dataset():
    train_dataset = h5py.File('week-2/data/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('week-2/data/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig

def load_flatten_dataset():

    x_train, y_train, x_test, y_test = load_dataset()
    shape_px = x_test.shape[0]

    print ("Number of training examples: m_train = " + str(x_train.shape[0]))
    print ("Number of testing examples: m_test = " + str(x_test.shape[0]))
    print ("Height/Width of each image: num_px = " + str(shape_px))
    print ("Each image is of size: (", str(shape_px), ", " , str(shape_px) ,", 3)")
    print ("x_train shape: ", str(x_train.shape))
    print ("y_train shape: ", str(y_train.shape))
    print ("x_test shape: ",str(x_test.shape))
    print ("y_test shape: ", str(y_test.shape))

    #flatten image
    x_train_flatten = x_train.reshape(x_train.shape[0], -1).T
    x_test_flatten = x_test.reshape(x_test.shape[0], -1).T

    print ("train_set_x_flatten shape: " + str(x_train_flatten.shape))
    print ("test_set_x_flatten shape: " + str(x_test_flatten.shape))
    print ("sanity check after reshaping: " + str(x_train_flatten[0:5,0]))

    x_train = x_train_flatten/255.
    x_test = x_test_flatten/255.

    return x_train, y_train, x_test, y_test

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def propagate(w, b , X, Y):

    m = X.shape[1]

    Z = np.dot(w.T, X) + b

    A = sigmoid(Z)

    cost = (-1/m) * np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1 - A)))

    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)

    return dw, db, cost

def optimize(w, b, X, Y, num_iterations, learning_rate = 0.01, print_cost = False):

    costs = []

    for i in range(num_iterations + 1):

        dw, db, cost = propagate(w, b, X, Y)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f " % (i, cost))

        costs.append(cost)

    return w, b , dw, db, costs


def predict(w, b, X):

    m = X.shape[1]
    predictions = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)

    for i in range(A.shape[1]):
        predictions[0, i] = 0 if A[0, i] <= 0.5 else 1

    return predictions


def model(x_train, y_train,
          x_test, y_test,
          num_iterations = 2000, learning_rate = 0.01, print_cost = False):

    w, b = np.zeros((x_train.shape[0], 1)), 0

    w, b , dw, db, costs = optimize(w, b, x_train, y_train, num_iterations, learning_rate, print_cost)

    prediction_test = predict(w, b, x_test)
    prediction_train = predict(w, b, x_train)

    return prediction_test, prediction_train, costs

def evaluate(y_label, y_prediction):
    m_test = y_prediction.shape[1]
    corrects = 0
    for i in range(m_test):
        if y_prediction[0,i] == y_label[0, i]:
            corrects+= 1

    return (corrects * 100 ) / m_test


x_train, y_train, x_test, y_test = load_flatten_dataset()
prediction_test, prediction_train, costs = model(x_train, y_train, x_test, y_test,2000,0.0001,print_cost = True)
print('Train accuracy : %f' % evaluate(y_train, prediction_train))
print('Test accuracy : %f' % evaluate(y_train, prediction_test))
