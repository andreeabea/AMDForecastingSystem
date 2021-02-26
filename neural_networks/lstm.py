import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import matplotlib.pyplot as plt

from data_layer.build_dataset import split_data_lstm
import numpy as np


class Lstm:

    def __init__(self, nb_features=256, nb_sequences=1, dataX=None, dataY=None):
        self.nb_features = nb_features
        self.nb_sequences = nb_sequences

        self.trainX, self.trainY, self.validX, self.validY, self.testX, self.testY = split_data_lstm(dataX, dataY)

        # self.scaler = MinMaxScaler(feature_range=(0, 1))
        # self.trainX = self.scaler.fit_transform(self.trainX)
        # self.trainY = self.scaler.fit_transform(self.trainY)
        # self.validX = self.scaler.fit_transform(self.validX)
        # self.validY = self.scaler.fit_transform(self.validY)

        self.trainX = np.array(self.trainX)
        self.trainX = self.trainX.reshape(-1, len(self.trainX), self.nb_features)
        self.trainY = np.array(self.trainY)
        self.trainY = self.trainY.reshape(-1, len(self.trainY), self.nb_features)
        self.validX = np.array(self.validX)
        self.validX = self.validX.reshape(-1, len(self.validX), self.nb_features)
        self.validY = np.array(self.validY)
        self.validY = self.validY.reshape(-1, len(self.validY), self.nb_features)
        self.testX = np.array(self.testX)
        self.testX = self.testX.reshape(-1, len(self.testX), self.nb_features)
        self.testY = np.array(self.testY)
        self.testY = self.testY.reshape(-1, len(self.testY), self.nb_features)

        self.model = self.build_lstm()

    def build_lstm(self):
        #create and fit the multivariate LSTM network
        model = Sequential()
        model.add(LSTM(64, input_shape=(1, self.nb_features)))
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

        self.model.reset_states()

        #plt.savefig('../plots/loss-lstm.png')

    def evaluate_model(self):
        # evaluate model
        if len(self.testX) > 0:
            print("Evaluate on test data")
            results = self.model.evaluate(self.testX, self.testY)
            print("test loss:", results[0])
            print("Predict ...")
            test_predict = self.model.predict(self.testX, batch_size=1)
            #prediction = self.scaler.inverse_transform(test_predict)
            prediction = test_predict
            print("predicted value: ", prediction)
            print("testX : ", self.testX)
            print("testY: ", self.testY)
            return results[0], prediction
        else:
            return 1000, None


if __name__ == '__main__':
    lstm = Lstm()
    lstm.train()
