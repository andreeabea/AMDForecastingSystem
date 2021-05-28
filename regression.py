import numpy as np
import pandas as pd
#import shap
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR

from tslearn.svm import TimeSeriesSVR
from tslearn.utils import to_time_series_dataset

import config
import tensorflow as tf
import matplotlib.pyplot as plt

from data_handling.db_handler import DbHandler
from data_handling.feature_selection import FeatureSelector
from data_handling.timeseries_augmentation import TimeSeriesGenerator
from neural_networks.rnn import Rnn

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

    def train_test_val_split(self, dataX, dataY):
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
        clf = linear_model.Lasso(alpha=0.1)
        cv = KFold(n_splits=10)
        n_scores = cross_val_score(clf, X, Y, cv=cv, n_jobs=-1)
        print('Accuracy: ' + str(np.mean(n_scores)))

    def voting_regression(self, include_timestamp=False, previous_visits=1, features='exclude VA'):
        gbr = GradientBoostingRegressor()

        X, Y = self.gen.generate_timeseries(include_timestamp, previous_visits, features)
        X = X.reshape(-1, previous_visits, X.shape[1])

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

    def rnn_regression(self, include_timestamp=False, previous_visits=1, features='exclude VA',
                   nn_type='lstm', custom=True):
        X, Y = self.gen.generate_timeseries(include_timestamp, previous_visits, features)
        trainX, trainY, validX, validY, testX, testY = self.train_test_val_split(X, Y)

        rnn = Rnn(trainX, trainY, validX, validY, testX, testY, nn_type=nn_type, custom=custom)
        rnn.train()

        rnn.evaluate_model()

    # explainer = shap.DeepExplainer(lstm.model, trainX)
    # shap_values = explainer.shap_values(testX)
    #
    # shap.initjs()
    #
    # # plot the explanation of the first prediction
    # # Note the model is "multi-output" because it is rank-2 but only has one column
    # shap.force_plot(explainer.expected_value[0], shap_values[0][0])


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


if __name__ == '__main__':
    #DatasetBuilder.write_all_data_to_csv("idk.csv", datatype='numerical', include_timestamps=True)
    datatype = 'numerical'
    include_timestamps = True

    db_handler = DbHandler(datatype, include_timestamps)
    data = db_handler.get_data_from_csv()

    reg = TimeSeriesRegressor(data)

    #feature_selector = FeatureSelector(data, reg.gen)
    #feature_vector = feature_selector.lasso_feature_selector(include_timestamps)
    #feature_vector = feature_selector.rfe(datatype, include_timestamps)
    #feature_vector = [1, 2, 10, 14, 15, 16, 18, 20, 22, 23, 24, 43, 47, 57, 99, 101, 123, 149, 172, 174, 177, 199, 227, 234, 244, 257, 275, 279]
    #feature_vector = [11, 12, 14, 18, 20, 22, 23]
    # for numerical 2 previous 57.64
    #simple GRU 75.38; rmspe: 1.47
    #my lstm 73.7
    # gru resampled 73.94
    #to try: predict also the future timestamp!
    #reg.rnn_regression(include_timestamps, 2, [0, 1], 'lstm', custom=True)
    for i in range(1, 4):
        #reg.gradient_boosted_regression(include_timestamps, i, feature_vector)
        reg.rnn_regression(include_timestamps, i, 'exclude VA', 'gru', custom=True)
