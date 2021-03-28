from trendypy.trendy import Trendy
from tslearn.clustering import TimeSeriesKMeans
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.svm import TimeSeriesSVR, TimeSeriesSVC
from tslearn.utils import to_time_series_dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def dtw_kmeans_clustering(data, nb_clusters):
    sequences = data.groupby('ID').apply(pd.DataFrame.to_numpy).to_numpy().tolist()
    sequences = to_time_series_dataset(sequences)
    km = TimeSeriesKMeans(n_clusters=nb_clusters, metric="dtw")
    labels = km.fit_predict(sequences)

    compute_accuracy(data, labels)


def dtw_clustering(data, nb_clusters):
    sequences = data.groupby('ID').apply(pd.DataFrame.to_numpy).to_numpy().tolist()
    trendy = Trendy(n_clusters=nb_clusters)
    trendy.fit(sequences)
    labels = trendy.labels_

    compute_accuracy(data, labels)


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


def svr_regression(data):
    reg = TimeSeriesSVR(kernel="gak", gamma="auto")


def svc_classifier(data):
    svc = TimeSeriesSVC(kernel="gak", gamma="auto", probability=True)
    trainX, trainY, testX, testY = split_data(data)
    print(svc.fit(trainX, trainY).score(testX, testY))


def get_actual_labels(data):
    actual_labels = []
    for nb, group in data.groupby('ID'):
        actual_label = 0
        if group['VA'].iloc[0] >= group['VA'].iloc[group.shape[0] - 1]:
            actual_label = 1
        actual_labels.append(actual_label)

    return actual_labels


def compute_accuracy(data, labels):
    accuracy = 0
    actual_labels = get_actual_labels(data)
    nb_series = len(actual_labels)
    for i in range(nb_series):
        obtained_label = labels[i]
        if obtained_label == actual_labels[i]:
            accuracy += 1

    accuracy = float(accuracy) / nb_series
    print("clustering accuracy: " + str(accuracy))


# visualize cluster distribution
def clusters_distribution(labels):
    good = 0
    for l in labels:
        if l == 0:
            good += 1

    bad = len(labels) - good
    good = float(good) / len(labels)
    bad = float(bad) / len(labels)
    print(str(good) + " " + str(bad))

    labels = ['Good evolution', 'Bad evolution']
    percentages = [good, bad]
    plt.bar(labels, percentages)

    plt.show()
