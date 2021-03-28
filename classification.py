import numpy as np
import pandas as pd
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.shapelets import LearningShapelets
from tslearn.svm import TimeSeriesSVC
from tslearn.utils import to_time_series_dataset

from clustering import get_actual_labels


def split_data(data):
    labels = np.array(get_actual_labels(data))
    mask = np.random.rand(len(labels)) < 0.8
    trainY = labels[mask]
    testY = labels[~mask]

    i = 0
    trainX = []
    testX = []
    for nb, group in data.groupby('ID'):
        if mask[i] == 1:
            trainX.append(group)
        else:
            testX.append(group)
        i += 1

    trainX = pd.concat(trainX)
    testX = pd.concat(testX)

    trainX = trainX.groupby('ID').apply(pd.DataFrame.to_numpy).to_numpy().tolist()
    trainX = to_time_series_dataset(trainX)

    testX = testX.groupby('ID').apply(pd.DataFrame.to_numpy).to_numpy().tolist()
    testX = to_time_series_dataset(testX)

    return trainX, trainY, testX, testY


def knn_classifier(data, nb_neighbors):
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=nb_neighbors, metric="dtw")
    trainX, trainY, testX, testY = split_data(data)
    print(knn.fit(trainX, trainY).score(testX, testY))


# very low accuracy
def learning_shapelets_classifier(data):
    clf = LearningShapelets()
    trainX, trainY, testX, testY = split_data(data)
    print(clf.fit(trainX, trainY).score(testX, testY))


def svc_classifier(data):
    svc = TimeSeriesSVC(kernel="gak", gamma="auto", probability=True)
    trainX, trainY, testX, testY = split_data(data)
    print(svc.fit(trainX, trainY).score(testX, testY))
