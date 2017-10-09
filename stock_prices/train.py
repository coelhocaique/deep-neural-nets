import lstm
import dataset_loader as data
# import matplotlib.pyplot as plt
import datetime

print (datetime.datetime.now().time(), 'Start training ...')
score, accuracy, model_name, training_time = lstm.train(base_name='EUR_USD_STRATEGY_1_M1_SOFTMAX',
                                                        path_data='dataset/EUR_USD_M1_2017-07-01_TO_2017-09-30.csv',
                                                        timesteps=100,
                                                        batch_size=120,
                                                        epochs=3)
print (datetime.datetime.now().time(), 'finishes training ...')

# input_predict = data.load_data('dataset/EUR_USD_M1_2017-07-01_TO_2017-09-30.csv')
# real_labels = data.get_labels(input_predict)
# studied_data, full_batch_size = data.study_period(input_predict)
# reshaped_data, labels = data.fit_to_shape(studied_data, full_batch_size, 'predict')