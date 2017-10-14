import lstm
import datetime

print (datetime.datetime.now().time(), 'Start training ...')
score, accuracy, model_name, training_time = lstm.train(base_name='EUR_USD_STRATEGY_1_M1_SOFTMAX',
                                                        path_data='dataset/EUR_USD_M1_2005-01-01_TO_2017-06-30.csv',
                                                        timesteps=100,
                                                        batch_size=120,
                                                        epochs=20)
print (datetime.datetime.now().time(), 'finishes training ...')