'''
    This file implements a Recurrent Neural Network which is a Deep Learning method
        to predict stock prices.
    To eliminate Vanish and Exploding Gradient we use LSTM(Long Short-Term Memory), which uses
        a gating technique to deal with it

    This implementation follows the study provided on this paper <-paper here->
    The main goal is to use the predictions of this Neural Network integrated with
    Metatrade 5, which will have an automated bot to place orders by itself on this platform.

    Our main focus is to work with Mini-dollar(see docs) <-docs here ->

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


def load_model(path):
    model = None
    try:
        model = km.load_model(path)
    except ImportError:
        model = Sequential()

    return model

def load_data():
    return None

def normalised_data():
    return None

def build_model(units,input_dim):
    '''
    Input layer with 1 feature and 240 timesteps.
    LSTM layer with h = 25 hidden neurons and a dropout value of 0.16.
        This configuration yields 2,752 parameters for the LSTM,
        leading to a sensible number of approximately 93 parameters per observation.
    Output layer (dense layer) with two neurons and softmax activation function - a standard configuration.
    '''
    model = Sequential()
    model.add(LSTM(input_dim=input_dim,
                   return_sequences=True,
                   activation=Activation("sigmoid")))
    return None

def save_model(model,path):
    model_utils.save_model(model,path)
