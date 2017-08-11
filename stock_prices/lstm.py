'''
    This file implements a Recurrent Neural Network which is a Deep Learning method
        to predict stock prices.
    To eliminate Vanish and Exploding Gradient we use LSTM(Long Short-Term Memory), which uses
        a gating technique to deal with it

    This implementation follows the study provided on this paper <-paper here->
    The main goal is to use the predictions of this Neural Network integrated with
    Metatrade 5, which will have an automated bot to place orders by itself on this platform.

    Our final product will be the Metatrade bot, which will be sold in binary files.

    This idea was primary developed on our TCC(Graduation project) in 2017, which founded
    Mammon Trading  2017 - http://www.mammontrading.com.br/

    Author: Caique Dos Santos Coelho

    All Rights Reserved
'''
import time, os
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import keras.models as model_utils
import h5py
import keras
import dataset_loader as data
from custom_callbacks import EarlyStoppingLoadWeights

def load_model(path):
    model = Sequential()
    if path:
        try:
            model = model_utils.load_model(path)
        except ImportError:
            print model.to_json()

    return model

def save_model(model,path,save_to_json=True):
    if save_to_json:
        save_to_file(os.path.abspath('jsons/'+ path + '.json'),model.to_json())
    model_utils.save_model(model,os.path.abspath('models/'+ path + '.h5'))

def save_to_file(path,content):
    f = open(path,'w')
    f.write(content)
    f.close()

def build_model(input_shape,hidden_units=25,timesteps=240,path=None):
    '''
    Input layer with 1 feature and 240 timesteps.
    LSTM layer with h = 25 hidden neurons and a dropout value of 0.16.
        This configuration yields 2,752 parameters for the LSTM,
        leading to a sensible number of approximately 93 parameters per observation.
    Output layer (dense layer) with two neurons and softmax activation function - a standard configuration.
    '''
    model = load_model(path)
    model.add(LSTM(240,input_shape=input_shape,
                        return_sequences=True))

    model.add(LSTM(hidden_units,
                   dropout=float(pow(0.1,6)),
                   recurrent_dropout=float(pow(0.1,6)),
                   return_sequences=False))

    model.add(Dense(units=2,
                    activation='softmax'))

    #prepares the model for training
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

    return model

def train_model(model,train_data,
                train_labels,validate_data,validate_labels,filepath,
                epochs=1000,batch_size=32):
    #apply early stopping as a further mechanism to prevent overtting
    #after 10 epochs with no error descreasing, quit training
    early_stopping = EarlyStoppingLoadWeights(monitor='val_loss',
                                                path = filepath,
                                                 min_delta=0,
                                                 patience=0,
                                                 verbose=1,
                                                 mode='min')

    tensorboard = keras.callbacks.TensorBoard(log_dir='./tensorboard',
                                              histogram_freq=1,
                                              batch_size=batch_size,
                                              write_graph=True,
                                              write_images=True)

    checkpoint = keras.callbacks.ModelCheckpoint(filepath,
                                                 monitor='val_loss',
                                                 verbose=1,
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 mode='min',
                                                 period=1)

    start = time.time()
    model.fit(
        train_data,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data = (validate_data, validate_labels),
        callbacks=[early_stopping,checkpoint,tensorboard])

    training_time = int(time.time() - start) / 60
    print 'Training time: ',training_time,' minutes'
    return model,training_time

def evaluate_model(model,validate_data,validate_labels):
    score, accuracy = model.evaluate(validate_data,validate_labels)
    return score,accuracy

def predict(model):
    return model

def train(path_data = 'dataset/itau_2009-02-02_2017-03-06_closing_price.csv'):

    train_data,train_labels,validate_data,validate_labels = data.load(path_data)

    model = build_model(train_data.shape[1:])

    times = str(time.time())
    filepath = 'models/daily_weights_' + times +'.h5'

    model,training_time = train_model(model,train_data,
                                      train_labels,validate_data,
                                      validate_labels,filepath,
                                      batch_size=120)

    score, accuracy = evaluate_model(model,validate_data,validate_labels)

    model_name = 'daily_' + times + '_score_' + str(score * 100)

    save_model(model,model_name)

    return score,accuracy,model_name,training_time

def continue_traininig(model_name,path_data = 'dataset/itau_2009-02-02_2017-03-06_closing_price.csv'):
    model = load_model(model_name)

    train_data,train_labels,validate_data,validate_labels = data.load(path_data)

    model,training_time = train_model(model,train_data,
                                      train_labels,validate_data,
                                      validate_labels,batch_size=120)

    score, accuracy = evaluate_model(model,validate_data,validate_labels)

    save_model(model,model_name)

    return score,accuracy
