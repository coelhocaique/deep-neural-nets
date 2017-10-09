import lstm
import dataset_loader as data
import matplotlib.pyplot as plt

score, accuracy, model_name, training_time = lstm.train(base_name='EUR_USD_STRATEGY_1_M1',
                                                        path_data='dataset/EUR_USD_M1_2005-01-01_TO_2017-06-30.csv',
                                                        timesteps=100,
                                                        batch_size=32,
                                                        epochs=10)

# input_predict = data.load_data('dataset/predict.csv')
# real_labels = data.get_labels(input_predict)
# studied_data, full_batch_size = data.study_period(input_predict)
# reshaped_data, labels = data.fit_to_shape(studied_data, full_batch_size, 'predict')
