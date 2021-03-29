import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor
from sklearn.model_selection import KFold, cross_val_score

from tslearn.svm import TimeSeriesSVR
from tslearn.utils import to_time_series_dataset

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


def svr_regression(data):
    svr = TimeSeriesSVR(kernel="gak", gamma="auto")
    trainX, trainY, testX, testY = split_data(data)
    print(svr.fit(trainX, trainY).score(testX, testY))


def generate_timeseries(data, size=1):
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

    # split target visual acuity and move the future timestamp to X
    future_timestamp_array = Y[:, 2:]
    Y = Y[:, :1]
    X = X.reshape(-1, X.shape[1] * X.shape[2])
    if future_timestamp_array.shape[1] != 0:
        X = np.hstack((X, future_timestamp_array))

    return X, Y.reshape(-1)


def train_test_split(X, Y):
    mask = np.random.rand(len(X)) < 0.8
    trainX = X[mask]
    testX = X[~mask]
    trainY = Y[mask]
    testY = Y[~mask]

    return trainX.reshape(-1, trainX.shape[1]*trainX.shape[2]), trainY.reshape(-1), \
           testX.reshape(-1, testX.shape[1]*testX.shape[2]), testY.reshape(-1)


def voting_regression(data):
    X, Y = generate_timeseries(data)

    gbr = GradientBoostingRegressor()
    #abr = AdaBoostRegressor()
    #er = VotingRegressor([('gb', gbr), ('ab', abr)])
    #print(gbr.fit(trainX, trainY).score(testX, testY))
    cv = KFold(n_splits=10)
    n_scores = cross_val_score(gbr, X, Y, cv=cv, n_jobs=-1)
    print('Accuracy: ' + str(np.mean(n_scores)))
    plot_feature_importances(gbr.fit(X, Y), X.shape[1])


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


if __name__ == '__main__':
    data_builder = DatasetBuilder("./data_layer/DMLVAVcuID.xls")
    data_builder.get_visual_acuity_data()
    data_builder.format_timestamps()

    #data_builder.resample_time_series()
    print(data_builder.data)

    voting_regression(data_builder.data)
