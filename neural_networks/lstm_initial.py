import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt

from data_layer.build_dataset_v1 import split_data_lstm
import numpy as np

from experiments.visual_acuity_analysis import VisualAcuityAnalysis


class Lstm:

    def __init__(self, nb_features=257, nb_sequences=1, dataX=None, dataY=None):
        self.nb_features = nb_features
        self.nb_sequences = nb_sequences

        #self.scaler = Normalizer()

        self.scaler = None

        if dataX is not None and dataY is not None:
            self.trainX, self.trainY, self.validX, self.validY, self.testX, self.testY = split_data_lstm(self.scaler, None, dataX, dataY)
        else:
            va_analysis = VisualAcuityAnalysis()
            eyeData = va_analysis.get_va_df()

            self.trainX, self.trainY, self.validX, self.validY, self.testX, self.testY = split_data_lstm(None, eyeData, dataX, dataY)

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
        model.add(LSTM(128, input_shape=(self.nb_sequences, self.nb_features)))
        model.add(Dense(self.nb_features, activation="sigmoid"))

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
        if len(self.testX) > 0:
            print("Evaluate on test data")
            results = self.model.evaluate(self.testX, self.testY)
            print("test loss:", results)

            print("Predict ...")
            test_predict = self.model.predict(self.testX, batch_size=1)

            if self.scaler is not None:
                prediction = self.scaler.inverse_transform(test_predict)
            else:
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

    #lstm.model = tf.keras.models.load_model('D:\\Licenta\\licenta\\models\\lstm-va.h5')
    loss, prediction = lstm.evaluate_model()
