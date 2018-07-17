import numpy as np

class LogisticRegresion:

    def __init__(self, dim):
        self.w = np.zeros((dim, 1))
        self.b = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, m ,Y , A):
        return (-1/m) * np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1 - A)))

    def forward_pass(self, X):
        Z = np.dot(self.w.T, X) + self.b

        return self.sigmoid(Z)

    def backward_pass(self, m ,X, A, Y):
        dw = (1/m) * np.dot(X, (A - Y).T)
        db = (1/m) * np.sum(A - Y)

        return dw, db

    def propagate(self,X, Y):
        m = X.shape[1]
        A = self.forward_pass(X)
        cost = self.compute_cost(m ,Y , A)
        dw, db = self.backward_pass(m ,X, A, Y)

        return dw, db, cost

    def optimize(self, X, Y, num_iterations, learning_rate , print_cost):

        costs = []

        for i in range(num_iterations + 1):

            dw, db, cost = self.propagate(X, Y)

            self.w = self.w - learning_rate * dw
            self.b = self.b - learning_rate * db

            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f " % (i, cost))

            costs.append(cost)

        return dw, db, costs


    def predict(self, X):

        m = X.shape[1]
        predictions = np.zeros((1, m))
        self.w = self.w.reshape(X.shape[0], 1)

        A = self.forward_pass(X)

        for i in range(A.shape[1]):
            predictions[0, i] = 0 if A[0, i] <= 0.5 else 1

        return predictions


    def fit(self, x_train, y_train,
            x_test, y_test,
            num_iterations = 2000,
            learning_rate = 0.01,
            print_cost = False):

        dw, db, costs = self.optimize(x_train, y_train, num_iterations, learning_rate, print_cost)

        prediction_test = self.predict(x_test)
        prediction_train = self.predict(x_train)

        return prediction_train, prediction_test, costs

    def evaluate(self, y_label, y_prediction):
        m_test = y_prediction.shape[1]
        corrects = 0
        for i in range(m_test):
            if y_prediction[0,i] == y_label[0, i]:
                corrects+= 1

        return (corrects * 100 ) / m_test
