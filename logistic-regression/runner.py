from dataset_loader import load_dataset, flatten_dataset
from logistic_regression import model, evaluate

x_train, y_train, x_test, y_test = load_dataset()
x_train, x_test = flatten_dataset(x_train, x_test)
prediction_test, prediction_train, costs = model(x_train, y_train, x_test, y_test,2000,0.0001,print_cost = True)
print('Train accuracy : %f' % evaluate(y_train, prediction_train))
print('Test accuracy : %f' % evaluate(y_train, prediction_test))
