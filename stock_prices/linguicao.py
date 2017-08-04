import lstm
import time, os
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import keras.models as model_utils
import h5py
import keras

data = lstm.load_data('itau_2009-02-02_2017-03-06_closing_price.csv')
train_data,train_labels,validate_data,validate_labels = lstm.study_period(data)

model = Sequential()
print train_data.shape[1:]
print train_labels.shape
model.add(LSTM(240,input_shape=train_data.shape[1:],
                return_sequences=True))

model.add(LSTM(25,dropout=float(pow(0.1,6)),
               recurrent_dropout=float(pow(0.1,6)),
               return_sequences=False))

model.add(Dense(units=2,
                activation='softmax'))

#prepares the model for training
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss',
                                             min_delta=0,
                                            patience=10,
                                            verbose=0,
                                            mode='min')

tensorboard = keras.callbacks.TensorBoard(log_dir='./tensorboard',
                                         histogram_freq=1,
                                         batch_size=240,
                                         write_graph=True,
                                         write_images=True)
start = time.time()
model.fit(
    train_data,
    train_labels,
    batch_size=32,
    epochs=1000,
    validation_data = (validate_data, validate_labels),
    callbacks=[early_stopping,tensorboard])
print 'Trainign time: ',time.time() - start

score, accuracy = model.evaluate(validate_data,validate_labels)

print '%s: %.2f%% ' % ('Score',score*100)
print '%s: %.2f%% ' % ('Accuracy',accuracy*100)

path = str(time.time()) + "model"
f = open(path,'w')
f.write(model.to_json())
f.close()
model_utils.save_model(model,path + ".h5")
