import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import matplotlib.pyplot as plt

from build_dataset import split_data_lstm
import numpy as np

nb_features = 256
nb_sequences = 1

trainX, trainY, validX, validY, testX, testY = split_data_lstm()

scaler = MinMaxScaler(feature_range=(0, 1))
trainX = scaler.fit_transform(trainX)
trainY = scaler.fit_transform(trainY)
validX = scaler.fit_transform(validX)
validY = scaler.fit_transform(validY)

trainX = np.array(trainX)
trainX = trainX.reshape(-1, nb_sequences, nb_features)
trainY = np.array(trainY)
trainY = trainY.reshape(-1, nb_sequences, nb_features)
validX = np.array(validX)
validX = validX.reshape(-1, nb_sequences, nb_features)
validY = np.array(validY)
validY = validY.reshape(-1, nb_sequences, nb_features)

#create and fit the multivariate LSTM network
model = Sequential()
model.add(LSTM(256, input_shape=(nb_sequences, nb_features), return_sequences=True))
model.add(LSTM(256, return_sequences = True))
model.add(LSTM(256, return_sequences = True))
model.add(LSTM(128, return_sequences = False))
model.add(Dense(nb_features))#, activation="sigmoid"))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='gdrive/MyDrive/model.h5'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
]

my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='./lstm.h5'),
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
]

model.fit(trainX, trainY, epochs=50, batch_size=8, verbose=2,
          callbacks=my_callbacks,
          validation_data=(validX, validY))

plt.plot(model.history.history['loss'], label = 'Train loss', alpha = .5)
plt.plot(model.history.history['val_loss'], label = 'Val. loss', alpha = .5)
plt.title('Linear model loss')
plt.legend()
plt.show()

plt.savefig('./loss-lstm.png')
