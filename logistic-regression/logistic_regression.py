import numpy as np
import h5py

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
