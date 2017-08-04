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
    Company Founders:
      - Thiago
      - Caique
      - Gaby
      - Anderson

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


def load_model(path):
    model = None
    if path:
        try:
            model = model_utils.load_model(path)
        except ImportError:
            model = Sequential()
            print model.to_json()
    else:
        model = Sequential()

    return model

def load_data(path):
    f = open(path,'r').read()
    return f.split('\n')

def save_model(model,path,save_to_json=True):
    if save_to_json:
        f = open(path,'w')
        f.write(model.to_json())
        f.close()
    model_utils.save_model(model,path)

def build_model(units=25,input_dim=1,output_dim=240,path=None):
    '''
    Input layer with 1 feature and 240 timesteps.
    LSTM layer with h = 25 hidden neurons and a dropout value of 0.16.
        This configuration yields 2,752 parameters for the LSTM,
        leading to a sensible number of approximately 93 parameters per observation.
    Output layer (dense layer) with two neurons and softmax activation function - a standard configuration.
    '''
    model = load_model(path)
    model.add(LSTM(units=input_dim,
                   input_shape = (1760,240),
                   return_sequences=True))

    model.add(LSTM(units=units,
                   dropout=float(pow(0.1,6)),
                   return_sequences=False))

    model.add(Dense(units=2,
                    output_dim=1,
                    activation='softmax'))

    #prepares the model for training
    model.compile(loss='mse', optimizer='rmsprop',metrics=['accuracy'])

    return model

def fit_to_shape(data,full_batch_size,timesteps=240,input_dim=1):
    labels = []
    sequences = []

    for i in range(full_batch_size):
        #the paper extracts the cross-section median, but it not applied for us
        #we just extracts the median
        sequence = np.array(data[i:timesteps + i])
        mean = np.mean(sequence)
        if data[timesteps + i] > mean:
            label = 1
        else:
            label = 0
        labels.append(label)
        sequences.append(sequence)

    labels = np.array(labels)
    reshaped_data = np.empty((full_batch_size,timesteps,input_dim))
    for i in range(len(sequences)):
        reshaped_data[i] = sequences[i].reshape(len(sequences[i]),1)
    #75% is for training, 25% validating
    train_index = int(full_batch_size * 0.75)
    train_data,train_labels = reshaped_data[:train_index],labels[:train_index]
    validate_data,validate_labels =reshaped_data[train_index:],labels[train_index:]

    return train_data,train_labels,validate_data,validate_labels


def study_period(data,timesteps = 240,m = 1,input_dim=1):
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
    full_batch_size = len(profit) - timesteps

    return fit_to_shape(returns,full_batch_size)

def train_model(model,train_data,train_labels,
                epochs=1000,batch_size=240):
    #apply early stopping as a further mechanism to prevent overtting
    #after 10 epochs with no error descreasing, quit training
    early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 min_delta=0,
                                                patience=10,
                                                verbose=0,
                                                mode='min')

    tensorboard = keras.callbacks.TensorBoard(log_dir='./tensorboard',
                                             histogram_freq=1,
                                             batch_size=batch_size,
                                             write_graph=True,
                                             write_images=True)
    start = time.time()
    model.fit(
        train_data,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.05,
       callbacks=[early_stopping,tensorboard])
    print 'Trainign time: ',(time.time() - start) * 60,'minutes'
    return model

def evaluate_model(model,validate_data,validate_labels):
    score, accuracy = model.evaluate(validate_data,validate_labels)
    return score,accuracy

def predict(model):
    return model

def study_predictions(predictions):
    return None

def generate_sequences(first_train=True,
                        path=None,
                        path_data = 'itau_2009-02-02_2017-03-06_closing_price.csv'):
    data = load_data(path_data)

    model = build_model(path)

    train_data,train_labels,validate_data,validate_labels = study_period(data)
    print train_data.shape
    #model = train_model(model,train_data,train_labels,epochs = 1)

    #km.save_model(model,os.path.abspath("model.h5"))

    return None #evaluate_model(model,validate_data,validate_labels)


#path = os.path.abspath(str(time.time()) +"_model.h5")
#generate_sequences()
#
# print '%s: %.2f%% ' % ('Score',score*100)
# print '%s: %.2f%% ' % ('Accuracy',accuracy*100)
