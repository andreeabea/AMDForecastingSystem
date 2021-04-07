import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import numpy as np


class Lstm:

    def __init__(self, trainX, trainY, validX, validY, testX, testY, timesteps=1, nb_features=23):
        self.nb_sequences = 1
        self.timesteps = timesteps
        if trainX is not None and trainY is not None and validX is not None \
                and validY is not None and testX is not None and testY is not None:
            self.trainX = trainX
            self.trainY = trainY
            self.validX = validX
            self.validY = validY
            self.testX = testX
            self.testY = testY
            self.nb_features = trainX.shape[1]

            self.trainX = self.trainX.reshape(-1, timesteps, self.nb_features)
            self.validX = self.validX.reshape(-1, timesteps, self.nb_features)
            self.testX = self.testX.reshape(-1, timesteps, self.nb_features)
        else:
            self.nb_features = nb_features

        self.model = self.build_lstm()

    def build_lstm(self):
        #create and fit the multivariate LSTM network
        model = Sequential()
        model.add(LSTM(128, input_shape=(self.timesteps, self.nb_features)))
        model.add(Dense(1, activation="sigmoid"))

        return model

    @staticmethod
    def rmspe(y_true, y_pred):
        pct_var = (y_true - y_pred) / (y_true + 0.0000001)
        return K.sqrt(K.mean(K.square(pct_var)))

    def train(self):
        # MAE performed better than MSE or others
        self.model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

        my_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath='../models/lstm-va.h5'),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        ]

        self.model.fit(self.trainX, self.trainY, epochs=500, batch_size=8, verbose=2,
                       callbacks=my_callbacks, validation_data=(self.validX, self.validY))

        plt.plot(self.model.history.history['loss'], label='Train loss', alpha=.5)
        plt.plot(self.model.history.history['val_loss'], label='Val. loss', alpha=.5)
        plt.title('Linear model loss')
        plt.legend()
        plt.show()

        #plt.savefig('../plots/loss-lstm.png')

    def evaluate_model(self):
        # evaluate model
        print("Evaluate on test data")
        results = self.model.evaluate(self.testX, self.testY)
        print("test loss:", results)

        print("Predict ...")
        prediction = self.model.predict(np.array([self.testX[0]]), batch_size=1)

        print("predicted value: ", prediction)
        print("testX : ", [self.testX[0]])
        print("testY: ", self.testY[0])
