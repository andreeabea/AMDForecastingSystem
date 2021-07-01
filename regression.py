import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.svm import SVR

from tslearn.svm import TimeSeriesSVR
from tslearn.utils import to_time_series_dataset

import config
import tensorflow as tf
import matplotlib.pyplot as plt

from data_handling.db_handler import DbHandler
from data_handling.feature_selection import FeatureSelector
from data_handling.timeseries_augmentation import TimeSeriesGenerator
from neural_networks.cnn import Cnn
from neural_networks.rnn import Rnn

from tensorflow.keras.layers import Dense, concatenate, Dropout
from tensorflow.keras.models import Model

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:,.2f}'.format


class TimeSeriesRegressor:

    def __init__(self, data):
        self.data = data
        self.gen = TimeSeriesGenerator(data)

    def split_data(self):
        split_index = len(self.data.groupby('ID')) * 0.8
        i = 0
        trainX = []
        trainY = []
        testX = []
        testY = []
        for nb, group in self.data.groupby('ID'):
            j = group.index.size - 1
            to_predict = group.iloc[j]['VA']
            if i < split_index:
                trainX.append(group.head(j))
                trainY.append(to_predict)
            else:
                testX.append(group.head(j))
                testY.append(to_predict)
            i += 1

        trainX = pd.concat(trainX)
        testX = pd.concat(testX)

        trainX = trainX.groupby('ID').apply(pd.DataFrame.to_numpy).to_numpy().tolist()
        trainX = to_time_series_dataset(trainX)
        trainX = np.array(trainX)

        testX = testX.groupby('ID').apply(pd.DataFrame.to_numpy).to_numpy().tolist()
        testX = to_time_series_dataset(testX)

        trainY = np.array(trainY)
        testY = np.array(testY)

        return trainX, trainY, testX, testY

    def svr_tslearn(self):
        svr = TimeSeriesSVR(kernel="gak", gamma="auto")
        trainX, trainY, testX, testY = self.split_data()
        print(svr.fit(trainX, trainY).score(testX, testY))

    @staticmethod
    def train_test_val_split(dataX, dataY):
        # compute the training split
        i = int(len(dataX) * config.TRAIN_SPLIT)
        trainX = dataX[:i]
        trainY = dataY[:i]

        # obtain the validation and testing splits
        valid = int(len(dataX) * config.VAL_SPLIT)
        validX = dataX[i:i + valid]
        validY = dataY[i:i + valid]

        testX = dataX[i + valid:]
        testY = dataY[i + valid:]

        # return trainX.reshape(-1, trainX.shape[1]*trainX.shape[2]), trainY.reshape(-1), \
        #        testX.reshape(-1, testX.shape[1]*testX.shape[2]), testY.reshape(-1)
        return trainX, trainY, validX, validY, testX, testY

    def svr_regression(self, include_timestamp=False, previous_visits=1, features='all'):
        X, Y = self.gen.generate_timeseries(include_timestamp, previous_visits, features)

        svr = SVR()
        cv = KFold(n_splits=10)
        n_scores = cross_val_score(svr, X, Y, cv=cv, n_jobs=-1)
        print('R^2: ' + str(np.mean(n_scores)))

    def gradient_boosted_regression(self, include_timestamp=False, previous_visits=1, features='all'):
        X, Y = self.gen.generate_timeseries(include_timestamp, previous_visits, features)
        gbr = GradientBoostingRegressor()

        cv = KFold(n_splits=10)
        n_scores = cross_val_score(gbr, X, Y, cv=cv, n_jobs=-1)
        print('R^2: ' + str(np.mean(n_scores)))
        fit = gbr.fit(X, Y)
        self.plot_feature_importances(fit, X.shape[1])

    def random_forest_regression(self, include_timestamp=False, previous_visits=1, features='all'):
        X, Y = self.gen.generate_timeseries(include_timestamp, previous_visits, features)
        gbr = RandomForestRegressor()

        cv = KFold(n_splits=10)
        n_scores = cross_val_score(gbr, X, Y, cv=cv, n_jobs=-1)
        print('R^2: ' + str(np.mean(n_scores)))
        fit = gbr.fit(X, Y)
        self.plot_feature_importances(fit, X.shape[1])

    def extratrees_regression(self, include_timestamp=False, previous_visits=1, features='all'):
        X, Y = self.gen.generate_timeseries(include_timestamp, previous_visits, features)
        gbr = ExtraTreesRegressor()

        cv = KFold(n_splits=10)
        n_scores = cross_val_score(gbr, X, Y, cv=cv, n_jobs=-1)
        print('R^2: ' + str(np.mean(n_scores)))
        fit = gbr.fit(X, Y)
        self.plot_feature_importances(fit, X.shape[1])

    def lasso_regression(self, include_timestamp=False, previous_visits=1, features='exclude VA'):
        X, Y = self.gen.generate_timeseries(include_timestamp, previous_visits, features)
        clf = linear_model.Lasso(alpha=0.0001)
        cv = KFold(n_splits=10)
        n_scores = cross_val_score(clf, X, Y, cv=cv, n_jobs=-1)
        print('R^2: ' + str(np.mean(n_scores)))

    def linear_regression(self, include_timestamp=False, previous_visits=1, features='exclude VA'):
        X, Y = self.gen.generate_timeseries(include_timestamp, previous_visits, features)
        clf = linear_model.LinearRegression()
        cv = KFold(n_splits=10)
        n_scores = cross_val_score(clf, X, Y, cv=cv, n_jobs=-1)
        print('R^2: ' + str(np.mean(n_scores)))

    def voting_regression_v1(self, previous_visits=1, features='exclude VA'):
        gbr = GradientBoostingRegressor()

        X, Y = self.gen.generate_timeseries(size=previous_visits, features=features)
        X = X.reshape(-1, previous_visits, X.shape[1]//previous_visits)

        def build_lstm():
            lstm = Rnn(None, None, None, None, None, None, timesteps=previous_visits, nb_features=X.shape[2])
            lstm.model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
            return lstm.model

        keras_lstm = tf.keras.wrappers.scikit_learn.KerasRegressor(build_lstm, epochs=50, verbose=False)
        keras_lstm._estimator_type = 'regressor'

        vr = VotingRegressor([('gb', gbr), ('lstm', keras_lstm)])
        cv = KFold(n_splits=10)
        n_scores = cross_val_score(vr, X, Y, cv=cv, n_jobs=-1)
        print('R^2: ' + str(np.mean(n_scores)))

    @staticmethod
    def plot_feature_importances(regressor, nb_features):
        feature_names = list(range(0, nb_features))
        sorted_ids = np.argsort(regressor.feature_importances_)
        pos = np.arange(sorted_ids.shape[0]) + .5
        fig = plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.barh(pos, regressor.feature_importances_[sorted_ids], align='center')
        plt.yticks(pos, np.array(feature_names)[sorted_ids])
        plt.title('Feature Importance')
        plt.show()

    def rnn_regression_cv(self, include_timestamp=False, previous_visits=1, features='exclude VA',
                   nn_type='lstm', custom=True):
        X, Y = self.gen.generate_timeseries(include_timestamp, previous_visits, features)

        nb_folds = 10
        rand = 42
        kfold = KFold(n_splits=nb_folds, shuffle=True, random_state=rand)
        r2_score = 0
        mae_score = 0
        mse_score = 0
        rmse_score = 0
        rmspe_score = 0
        max_r2 = 0

        for train, test in kfold.split(X, Y):
            trainX, validX, trainY, validY = train_test_split(X[train], Y[train], test_size = 0.3, random_state = rand)
            rnn = Rnn(trainX, trainY, validX, validY, X[test], Y[test], nn_type=nn_type, custom=custom)
            rnn.train()

            mae, mse, rmse, r2, rmspe = rnn.evaluate_model()
            r2_score += r2
            mse_score += mse
            mae_score += mae
            rmse_score += rmse
            rmspe_score += rmspe
            input()
            if r2 > max_r2:
                max_r2 = r2

        print('Avg cross-validated MSE score: ' + str(float(mse_score / nb_folds)))
        print('Avg cross-validated MAE score: ' + str(float(mae_score / nb_folds)))
        print('Avg cross-validated RMSE score: ' + str(float(rmse_score / nb_folds)))
        print('Avg cross-validated R^2 score: ' + str(float(r2_score/nb_folds)))
        print('Best R^2 score: ' + str(max_r2))

    def rnn_regression(self, include_timestamp=False, previous_visits=1, features='exclude VA',
                   nn_type='lstm', custom=True):
        X, Y = self.gen.generate_timeseries(include_timestamp, previous_visits, features)
        trainX, trainY, validX, validY, testX, testY = self.train_test_val_split(X, Y)

        rnn = Rnn(trainX, trainY, validX, validY, testX, testY, nn_type=nn_type, custom=custom)
        rnn.train()

        rnn.evaluate_model()

    def lstm_ensemble_regression(self, include_timestamp=False, previous_visits=1, features='exclude VA'):
        X, Y = self.gen.generate_timeseries(include_timestamp, previous_visits, features)
        X = X.reshape(-1, previous_visits, int(X.shape[1]/previous_visits))

        def build_lstm():
            lstm = Rnn(None, None, None, None, None, None, timesteps=previous_visits, nb_features=X.shape[2])
            lstm.model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
            return lstm.model

        keras_lstm = tf.keras.wrappers.scikit_learn.KerasRegressor(build_lstm, epochs=30, verbose=False)
        keras_lstm._estimator_type = 'regressor'

        lstm_ensemble = AdaBoostRegressor(base_estimator=keras_lstm)
        cv = KFold(n_splits=10)
        n_scores = cross_val_score(lstm_ensemble, X, Y, cv=cv, n_jobs=-1)
        print('R^2: ' + str(np.mean(n_scores)))

    def rnn_ensemble_regression(self, include_timestamp=False, previous_visits=1, features='exclude VA'):
        X, Y = self.gen.generate_timeseries(include_timestamp, previous_visits, features)
        trainX, trainY, validX, validY, testX, testY = self.train_test_val_split(X, Y)

        members = []
        dependencies = {
            'root_mean_squared_error': Rnn.root_mean_squared_error
        }

        lstm = tf.keras.models.load_model("best_models/lstm-num-img-res-0.95.h5", custom_objects=dependencies)
        members.append(lstm)
        #lstm.summary()
        #lstm = Rnn(trainX, trainY, validX, validY, testX, testY, model=lstm)
        #lstm.evaluate_model()

        cnn = tf.keras.models.load_model("best_models/rnn-num-img-res-0.95.h5", custom_objects=dependencies)
        members.append(cnn)
        #cnn.summary()
        gru = tf.keras.models.load_model("models/gru2-numerical-woVA2.h5", custom_objects=dependencies)
        #members.append(gru)
        #gru = Rnn(trainX, trainY, validX, validY, testX, testY, model=gru)
        #gru.evaluate_model()

        # update all layers in all models to not be trainable
        for i in range(len(members)):
            model = members[i]
            #model.get_layer(name='bidirectional_input').name = 'ensemble_' + str(i + 1) + '_' + 'bidirectional_input'
            #model.inputs._name = 'ensemble_' + str(i + 1) + '_' + model.inputs.name
            #model.input.type_spec._name = 'ensemble_' + str(i + 1) + '_' + model.input.type_spec.name
            for layer in model.layers:
                # make not trainable
                layer.trainable = False
                # rename to avoid 'unique layer name' issue
                layer._name = 'ensemble_' + str(i + 1) + '_' + layer.name

        # define multi-headed input
        ensemble_visible = [model.input for model in members]
        # concatenate merge output from each model
        ensemble_outputs = [model.output for model in members]
        merge = concatenate(ensemble_outputs)
        output = Dense(1, activation='sigmoid')(merge)
        ensemble = Model(inputs=ensemble_visible, outputs=output)

        ensemble_rnn = Rnn(trainX, trainY, validX, validY, testX, testY, model=ensemble)

        new_trainX = [ensemble_rnn.trainX for _ in range(len(ensemble.input))]
        new_validX = [ensemble_rnn.validX for _ in range(len(ensemble.input))]
        new_testX = [ensemble_rnn.testX for _ in range(len(ensemble.input))]

        ensemble_rnn.trainX = new_trainX
        ensemble_rnn.validX = new_validX
        ensemble_rnn.testX = new_testX
        ensemble_rnn.train()
        ensemble_rnn.evaluate_model()

    def voting_regression(self, include_timestamp=False, previous_visits=1, features='exclude VA'):
        X, Y = self.gen.generate_timeseries(include_timestamp, previous_visits, features)
        nb_folds = 10
        rand = 42
        kfold = KFold(n_splits=nb_folds, shuffle=True, random_state=rand)
        r2_score_cv = 0
        max_r2 = 0

        for train, test in kfold.split(X, Y):
            #_, _, _, _, testX, testY = self.train_test_val_split(X, Y)
            testX, testY = X[test], Y[test]
            testX = testX.reshape(-1, 1, testX.shape[1])

            members = []
            dependencies = {
                'root_mean_squared_error': Rnn.root_mean_squared_error
            }

            lstm = tf.keras.models.load_model("best_models/lstm-num-img-res-0.95.h5", custom_objects=dependencies)
            members.append(lstm)
            lstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error',
                                                                                     'mean_squared_error',
                                                                                     Rnn.root_mean_squared_error])
            lstm.evaluate(testX, testY)

            cnn = tf.keras.models.load_model("best_models/rnn-num-img-res-0.95.h5", custom_objects=dependencies)
            members.append(cnn)
            cnn.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error',
                                                                                     'mean_squared_error',
                                                                                     Rnn.root_mean_squared_error])
            cnn.evaluate(testX, testY)

        #gru = tf.keras.models.load_model("models/gru2-numerical-woVA2.h5", custom_objects=dependencies)
        #members.append(gru)
        #gru.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error',
        #                                                                         'mean_squared_error',
        #                                                                         Rnn.root_mean_squared_error])
        #gru.evaluate(testX, testY)

            lstm_prediction = lstm.predict(testX, batch_size=1)
            print("Compute R^2 ...")
            result = r2_score(testY, lstm_prediction)
            print(result)
            cnn_prediction = cnn.predict(testX, batch_size=1)
            print("Compute R^2 ...")
            result = r2_score(testY, cnn_prediction)
            print(result)
            #gru_prediction = gru.predict(testX, batch_size=1)

            predictions = np.array([lstm_prediction, cnn_prediction])#, gru_prediction])
            avg_prediction = np.mean(predictions, axis=0)

            testY = testY.reshape(testY.shape[0], 1)
            print("RMSPE: ")
            result = Rnn.rmspe(testY, avg_prediction)
            print(result)
            print("Compute R^2 ...")
            r2 = r2_score(testY, avg_prediction)
            print(r2)

            r2_score_cv += r2
            input()
            if r2 > max_r2:
                max_r2 = r2

        print('Avg cross-validated R^2 score: ' + str(float(r2_score_cv / nb_folds)))


if __name__ == '__main__':
    #DatasetBuilder.write_all_data_to_csv("idk.csv", datatype='numerical', include_timestamps=True)
    datatype = 'numerical'
    include_timestamps = False

    db_handler = DbHandler(datatype, include_timestamps)
    data = db_handler.get_data_from_csv()

    reg = TimeSeriesRegressor(data)

    #feature_selector = FeatureSelector(data, reg.gen)
    #feature_vector = feature_selector.lasso_feature_selector(include_timestamps)
    #feature_vector = feature_selector.rfe(datatype, include_timestamps)
    #feature_vector = [1, 2, 10, 14, 15, 16, 18, 20, 22, 23, 24, 43, 47, 57, 99, 101, 123, 149, 172, 174, 177, 199, 227, 234, 244, 257, 275, 279]
    #feature_vector = [11, 12, 14, 18, 20, 22, 23]
    #feature_vector=[5, 11, 12, 15, 16, 17, 20, 22, 23, 24, 83, 101, 120, 138, 168, 171, 172, 179, 180, 190, 199, 212, 227, 230, 244, 252, 257, 270]
    # for numerical 2 previous 57.64
    #simple GRU 75.38; rmspe: 1.47
    #my lstm 73.7
    # gru resampled 73.94
    #reg.rnn_regression(include_timestamps, 2, [0, 1], 'lstm', custom=True)
    #for i in range(1, 4):
        #reg.gradient_boosted_regression(include_timestamps, i, 'exclude VA')
    #feature_vector = [1, 20, 22, 24, 37, 56, 76, 117, 190, 211, 244]
    #feature_vector = [1, 20, 21, 22,24, 76, 117, 244, 275]
    reg.rnn_regression_cv(include_timestamps, previous_visits=3,
                        features=[0, 1],
                        nn_type='lstm', custom=True)
    #reg.voting_regression(include_timestamps, previous_visits=2, features='exclude VA')