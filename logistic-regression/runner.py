from dataset_loader import load_dataset, flatten_dataset
from logistic_regression import LogisticRegresion

x_train, y_train, x_test, y_test = load_dataset()
x_train, x_test = flatten_dataset(x_train, x_test)

model = LogisticRegresion(x_train.shape[0])

prediction_train, prediction_test, costs = model.fit(x_train, y_train,
                                                   x_test, y_test,
                                                   num_iterations = 20000,
                                                   learning_rate = 0.01,
                                                   print_cost = True)

print('Train accuracy : %f' % model.evaluate(y_train, prediction_train))
print('Test accuracy : %f' % model.evaluate(y_test, prediction_test))
