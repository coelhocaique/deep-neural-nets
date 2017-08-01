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
    Mammon Trading © 2017 - http://www.mammontrading.com.br/

    Company Founders:
      - Thiago
      - Caique
      - Gaby
      - Anderson

    Author: Caique Dos Santos Coelho

    All Rights Reserved ®
'''
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import keras.models as model_utils
import h5py
import keras as keras


def load_model(path):
    model = None
    try:
        model = model_utils.load_model(path)
    except ImportError:
        model = Sequential()

    return model

def load_data(path):
    f = open(path,'r').read()
    return f.split('\n')

def study_period(data,timesteps = 240,m = 1):
    '''
        input_shape = (batch_size, timesteps, input_dim)`
        input_shape = [N ,240,1]
    '''
    #calculate return
    profit = []
    for i in range(1,len(data)):
        try:
            current_return = (float(data[i])/float(data[i-m])) - 1
            profit.append(current_return)
        except ValueError,e:
            print "error",e,"on line",i

    #standardize return
    returns = np.array(profit)
    mean = np.mean(returns)
    standard_deviation =  np.std(returns)
    returns = [(ret - mean) / standard_deviation for ret in returns]

    #75% of the data is for training, the other 25% is for validation
    trainig_data,training_labels = 0,0
    validation_data,validation_labels = 0,0




def normalised_data():
    return None

def predict(model):
    return model

def study_predictions(predictions):
    return None

def train_model(model,epochs=1000,batch_size=240):
    #apply early stopping as a further mechanism to prevent overtting
    #after 10 epochs with no error descreasing, quit training
    early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 min_delta=0,
                                                patience=10,
                                                verbose=0,
                                                mode='min')

    tensorboard = keras.callbacks.TensorBoard(log_dir='./tensorboard',
                                             histogram_freq=1,
                                             batch_size=batch_size
                                             write_graph=True,
                                             write_images=True)
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.05,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping,tensorboard])

    return model

def build_model(units=25,input_dim=1,output_dim=240,path=None):
    '''
    Input layer with 1 feature and 240 timesteps.
    LSTM layer with h = 25 hidden neurons and a dropout value of 0.16.
        This configuration yields 2,752 parameters for the LSTM,
        leading to a sensible number of approximately 93 parameters per observation.
    Output layer (dense layer) with two neurons and softmax activation function - a standard configuration.
    '''
    model = load_model(path)
    model.add(LSTM(units=input_dim
                   input_dim=input_dim,
                   output_dim=output_dim
                   return_sequences=True))

    model.add(LSTM(units=units,
                   return_sequences=False,
                   dropout=float(pow(0.1,6)),
                   use_bias=True))

    model.add(Dense(units=2,
                    output_dim=1,
                    activation='softmax'))

    #prepares the model for training
    model.compile(loss='mse', optimizer='rmsprop',metrics=['accuracy'])

    return model

def save_model(model,path,save_to_json = True):
    if save_to_json:
        f = open(path,'w')
        f.write(model.to_json())
        f.close()
    model_utils.save_model(model,path)

def generate_sequences(first_train=True,path=None,path_data = 'itau_2009-02-02_2017-03-06_closing_price.csv'):
    model=None
    #load data and apply study for training lstm
    #incomplete call && function
    data = load_data(path_data)

    trainig_data,training_labels,validation_data,validation_labels = study_period(data)

    model = build_model()

    #missing parameters
    model=train_model(model)

    predictions=predict(model)
    #km.save_model(model,os.path.abspath("model.h5"))

    output_predictions=study_predictions(predictions)

    return output_predictions
