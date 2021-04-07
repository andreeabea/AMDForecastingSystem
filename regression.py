import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from tslearn.svm import TimeSeriesSVR
from tslearn.utils import to_time_series_dataset

import config
from data_layer.build_dataset import DatasetBuilder
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt


def split_data(data):
    split_index = len(data.groupby('ID')) * 0.8
    i = 0
    trainX = []
    trainY = []
    testX = []
    testY = []
    for nb, group in data.groupby('ID'):
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


def train_test_split(X, Y):
    mask = np.random.rand(len(X)) < 0.8
    trainX = X[mask]
    testX = X[~mask]
    trainY = Y[mask]
    testY = Y[~mask]

    return trainX.reshape(-1, trainX.shape[1]*trainX.shape[2]), trainY.reshape(-1), \
           testX.reshape(-1, testX.shape[1]*testX.shape[2]), testY.reshape(-1)


def svr_regression_tslearn(data):
    svr = TimeSeriesSVR(kernel="gak", gamma="auto")
    trainX, trainY, testX, testY = split_data(data)
    print(svr.fit(trainX, trainY).score(testX, testY))


def generate_timeseries(data, include_timestamp=False, size=1, features='all'):
    X = []
    Y = []

    sequences = data.groupby('ID').apply(pd.DataFrame.to_numpy).to_numpy().tolist()
    for i in range(len(sequences)):
        if len(sequences[i]) > size:
            generator = TimeseriesGenerator(sequences[i], sequences[i], length=size)
            x, y = generator[0]
            X.append(x)
            Y.append(y)

    X = np.vstack(X)
    Y = np.vstack(Y)

    # select most important features: 45% acc?
    feature_list = [22, 23, 17, 21, 11, 12, 7, 20, 16]

    if include_timestamp:
        # update the timestamps
        for i in range(len(X)):
            min_timestamp = X[i, 0, X.shape[2]-1]
            X[i, :, X.shape[2]-1] = np.subtract(X[i, :, X.shape[2]-1], min_timestamp)
            Y[i, Y.shape[1]-1] = Y[i, Y.shape[1]-1] - min_timestamp
        # split target visual acuity and move the future timestamp to X
        future_timestamp_array = Y[:, Y.shape[1] - 1:]
        # append the timestamp to the feature list
        feature_list.append(24)
    Y = Y[:, :1]
    if features is not 'all':
        X = X[:, :, feature_list]
    X = X.reshape(-1, X.shape[1] * X.shape[2])
    if include_timestamp and future_timestamp_array.shape[1] != 0:
        X = np.hstack((X, future_timestamp_array))

    return X, Y.reshape(-1)


def svr_regression(data):
    X, Y = generate_timeseries(data, False)

    gbr = SVR()
    cv = KFold(n_splits=10)
    n_scores = cross_val_score(gbr, X, Y, cv=cv, n_jobs=-1)
    print('Accuracy: ' + str(np.mean(n_scores)))
    plot_feature_importances(gbr.fit(X, Y), X.shape[1])


def gradient_boosted_regression(data, include_timestamp=False, previous_visits=1, features='all'):
    X, Y = generate_timeseries(data, include_timestamp, previous_visits, features)

    gbr = GradientBoostingRegressor()
    cv = KFold(n_splits=10)
    n_scores = cross_val_score(gbr, X, Y, cv=cv, n_jobs=-1)
    print('Accuracy: ' + str(np.mean(n_scores)))
    plot_feature_importances(gbr.fit(X, Y), X.shape[1])


def extratrees_regression(data, include_timestamp=False):
    X, Y = generate_timeseries(data, include_timestamp)

    gbr = ExtraTreesRegressor()
    cv = KFold(n_splits=10)
    n_scores = cross_val_score(gbr, X, Y, cv=cv, n_jobs=-1)
    print('Accuracy: ' + str(np.mean(n_scores)))
    plot_feature_importances(gbr.fit(X, Y), X.shape[1])


def lasso_regression(data, include_timestamp=False):
    X, Y = generate_timeseries(data, include_timestamp)
    clf = linear_model.Lasso(alpha=0.1)
    cv = KFold(n_splits=10)
    n_scores = cross_val_score(clf, X, Y, cv=cv, n_jobs=-1)
    print('Accuracy: ' + str(np.mean(n_scores)))


def lstm_ensemble_regression(data):
    abr = AdaBoostRegressor()
    #er = VotingRegressor([('gb', gbr), ('ab', abr)])
    #print(gbr.fit(trainX, trainY).score(testX, testY))


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


def lstm_ensemble_regression():
    return None


if __name__ == '__main__':
    #DatasetBuilder.write_all_data_to_csv("all_data.csv", include_timestamps=True)
    include_timestamps = True
    if include_timestamps:
        data = pd.read_csv("all_data.csv", index_col=['ID', 'Date'], parse_dates=True)
    else:
        data = pd.read_csv("all_data_resampled.csv", index_col=['ID', 'Date'], parse_dates=True)
    gradient_boosted_regression(data, include_timestamps, 1, 'not all')
