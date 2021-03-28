import numpy as np
import pandas as pd

from tslearn.svm import TimeSeriesSVR
from tslearn.utils import to_time_series_dataset


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

    testX = testX.groupby('ID').apply(pd.DataFrame.to_numpy).to_numpy().tolist()
    testX = to_time_series_dataset(testX)

    trainY = np.array(trainY)
    testY = np.array(testY)

    return trainX, trainY, testX, testY


def svr_regression(data):
    svr = TimeSeriesSVR(kernel="gak", gamma="auto")
    trainX, trainY, testX, testY = split_data(data)
    print(svr.fit(trainX, trainY).score(testX, testY))
