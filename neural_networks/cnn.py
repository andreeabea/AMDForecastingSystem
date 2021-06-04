import tensorflow as tf
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, AveragePooling1D, Bidirectional, Conv1D, Flatten, LSTM, ReLU, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt


class Cnn:

    def __init__(self, trainX, trainY, validX, validY, testX, testY, timesteps=1, nb_features=23,
                 custom=True, model=None, nb_labels=2):
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

        if model is not None:
            self.model = model
        elif custom:
            self.model = self.build_cnn(nb_labels)

        if nb_labels == 2:
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def build_cnn(self, nb_labels):
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=self.timesteps, activation='relu', input_shape=(self.timesteps, self.nb_features)))
        model.add(AveragePooling1D(pool_size=1))
        model.add(Bidirectional(LSTM(64)))
        model.add(ReLU())
        model.add(Dropout(0.1))
        #model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(nb_labels, activation='softmax'))

        return model

    def train(self):
        my_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath='../models/cnn.h5'),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ]

        self.model.fit(self.trainX, self.trainY, epochs=500, batch_size=8, verbose=2,
                       callbacks=my_callbacks, validation_data=(self.validX, self.validY))

        plt.plot(self.model.history.history['loss'], label='Train loss', alpha=.5)
        plt.plot(self.model.history.history['val_loss'], label='Val. loss', alpha=.5)
        plt.title('Linear model loss')
        plt.legend()
        plt.show()

        plt.plot(self.model.history.history['accuracy'], label='Train loss', alpha=.5)
        plt.plot(self.model.history.history['val_accuracy'], label='Val. loss', alpha=.5)
        plt.title('Linear model loss')
        plt.legend()
        plt.show()

        # plt.savefig('../plots/loss-lstm.png')

    def evaluate_model(self):
        # evaluate model
        print("Evaluate on test data")
        results = self.model.evaluate(self.testX, self.testY)
        print("test loss:", results)

        print("Predict ...")
        prediction = self.model.predict(self.testX, batch_size=1)

        #print("predicted value: ", prediction)
        #print("testX : ", self.testX)
        #print("testY: ", self.testY)

        y_test = []
        for i in range(len(self.testY)):
            for j in range(2):
              if self.testY[i][j] == 1:
                  y_test.append(j)

        y_pred = []
        for i in range(len(prediction)):
            if prediction[i][0] > prediction[i][1]:
                y_pred.append(0)
            else:
                y_pred.append(1)

        conf_matrix = tf.math.confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True)
        plt.show()
