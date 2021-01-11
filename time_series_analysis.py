import math
import os
from xml.dom import minidom

from sklearn.decomposition import PCA
from trendypy.trendy import Trendy
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

import config
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


CENTRAL_AVG_THICKNESS_FEATURE = "CentralThickness"
C0_THICKNESS_FEATURE = "AvgThickness"


def get_feature_sequences(feature_name):
    x = 256
    y = 256
    sequences = []

    for patient in os.scandir(config.ORIG_INPUT_DATASET):
        visits = len(list(os.scandir(patient)))
        #print(patient.name + " " + str(visits))
        if (visits <= 1):
            continue
        featurelist_left = []
        featurelist_right = []
        for visit in os.scandir(patient):
            #print(visit.name)
            for eye in os.scandir(visit):
                for file in os.scandir(eye):
                    if file.path.endswith(".xml"):
                        xmlfile = minidom.parse(file.path)
                        #patientID = xmlfile.getElementsByTagName('ID')[0].firstChild.nodeValue
                        feature = xmlfile.getElementsByTagName(feature_name)
                        if feature[0].firstChild is not None:
                            if "OS" in eye.name:
                                featurelist_left.append(float(feature[0].firstChild.nodeValue))
                            else:
                                featurelist_right.append(float(feature[0].firstChild.nodeValue))

        if len(featurelist_left) > 1:
            sequences.append(featurelist_left)
        if len(featurelist_right) > 1:
            sequences.append(featurelist_right)

    return sequences


def plot_sequences(sequences, labels):
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 10.5)

    plt.style.use('fivethirtyeight')
    cmap = ['red', 'blue']

    for i in range(len(sequences)):
        nb_visits = len(sequences[i])
        if nb_visits > 2:
            visits = []
            for visit in range(nb_visits):
                visits.append(visit)
            label = labels[i]
            color = cmap[label]
            plt.plot(visits, sequences[i], color=color, alpha=0.5)
    plt.show()


def dtw_kmeans_clustering(sequences):
    trendy = Trendy(n_clusters=2)
    trendy.fit(sequences)
    labels = trendy.labels_
    # sequences = to_time_series_dataset(sequences)
    # km = TimeSeriesKMeans(n_clusters=2, metric="dtw")
    # labels = km.fit_predict(sequences)
    return labels


def find_correlation(central_thickness, avg_thickness):
    correlation = 0
    nb_timeseries = len(central_thickness)
    valid_series = 0
    for i in range(nb_timeseries):
        #plt.scatter(central_thickness[i], avg_thickness[i])
        #plt.show()
        if len(central_thickness[i]) == len(avg_thickness[i]):
            r, p = pearsonr(central_thickness[i], avg_thickness[i])
            if math.isnan(r) == False:
                correlation += r
                valid_series += 1
    correlation = float(correlation)/valid_series
    print("average correlation: " + str(correlation))


sequences1 = get_feature_sequences(CENTRAL_AVG_THICKNESS_FEATURE)
sequences2 = get_feature_sequences(C0_THICKNESS_FEATURE)
find_correlation(sequences1, sequences2)
#labels = dtw_kmeans_clustering(sequences)
#plot_sequences(sequences, labels)
