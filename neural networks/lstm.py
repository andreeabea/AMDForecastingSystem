import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import matplotlib.pyplot as plt

from data_layer.build_dataset import split_data_lstm
import numpy as np


class Lstm:

    def __init__(self, nb_features=256, nb_sequences=1):
        self.nb_features = nb_features
        self.nb_sequences = nb_sequences

        self.trainX, self.trainY, self.validX, self.validY, self.testX, self.testY = split_data_lstm()

        scaler = MinMaxScaler(feature_range=(0, 1))
        self.trainX = scaler.fit_transform(self.trainX)
        self.trainY = scaler.fit_transform(self.trainY)
        self.validX = scaler.fit_transform(self.validX)
        self.validY = scaler.fit_transform(self.validY)

        self.trainX = np.array(self.trainX)
        self.trainX = self.trainX.reshape(-1, self.nb_sequences, self.nb_features)
        self.trainY = np.array(self.trainY)
        self.trainY = self.trainY.reshape(-1, self.nb_sequences, self.nb_features)
        self.validX = np.array(self.validX)
        self.validX = self.validX.reshape(-1, self.nb_sequences, self.nb_features)
        self.validY = np.array(self.validY)
        self.validY = self.validY.reshape(-1, self.nb_sequences, self.nb_features)

        self.model = self.build_multivariate_lstm()

    def build_multivariate_lstm(self):
        #create and fit the multivariate LSTM network
        model = Sequential()
        model.add(LSTM(256, input_shape=(self.nb_sequences, self.nb_features)))
        model.add(Dense(self.nb_features))

        return model

    def train(self):
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

        my_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath='../models/lstm.h5'),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        ]

        self.model.fit(self.trainX, self.trainY, epochs=50, batch_size=8, verbose=2,
                       callbacks=my_callbacks, validation_data=(self.validX, self.validY))

        plt.plot(self.model.history.history['loss'], label='Train loss', alpha=.5)
        plt.plot(self.model.history.history['val_loss'], label='Val. loss', alpha=.5)
        plt.title('Linear model loss')
        plt.legend()
        plt.show()

        plt.savefig('../plots/loss-lstm.png')


if __name__ == '__main__':
    lstm = Lstm()
    lstm.train()
