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

from data_layer.build_dataset import create_sequences
from neural_networks.lstm import Lstm
from sklearn import linear_model

from visual_acuity_analysis import VisualAcuityAnalysis


class RetinaFeatureAnalysis:

    CENTRAL_AVG_THICKNESS_FEATURE = "CentralThickness"
    THICKNESS_FEATURE = "AvgThickness"

    def get_feature_sequences(self, feature_name, zone=0):
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
                            if feature[zone].firstChild is not None:
                                if "OS" in eye.name:
                                    featurelist_left.append(float(feature[zone].firstChild.nodeValue))
                                else:
                                    featurelist_right.append(float(feature[zone].firstChild.nodeValue))

            if len(featurelist_left) > 1:
                sequences.append(featurelist_left)
            if len(featurelist_right) > 1:
                sequences.append(featurelist_right)

        return sequences


    def plot_sequences(self, sequences, labels):
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

    def dtw_kmeans_clustering(self, sequences):
        sequences = to_time_series_dataset(sequences)
        km = TimeSeriesKMeans(n_clusters=2, metric="dtw")
        labels = km.fit_predict(sequences)

        return labels

    def find_correlation(self, feature1, feature2):
        correlation = 0
        valid_series = 0
        len_feature1 = len(feature1)
        len_feature2 = len(feature2)
        i = 0
        j = 0
        while i < len_feature1 and j < len_feature2:
            if len(feature1[i]) == len(feature2[j]):
                plt.scatter(feature1[i], feature2[i])
                r, p = pearsonr(feature1[i], feature2[j])
                if math.isnan(r) == False:
                    correlation += r
                    valid_series += 1
                i += 1
                j += 1
            else:
                if len_feature1 < len_feature2:
                    del feature2[j]
                else:
                    del feature1[i]
                len_feature1 = len(feature1)
                len_feature2 = len(feature2)
        plt.show()
        correlation = float(correlation)/valid_series
        print("average correlation: " + str(correlation))


if __name__ == '__main__':
    va_analysis = VisualAcuityAnalysis()
    eyeData = va_analysis.get_va_df()

    retina_analysis = RetinaFeatureAnalysis()

    sequences1 = retina_analysis.get_feature_sequences(RetinaFeatureAnalysis.THICKNESS_FEATURE, 3)
    sequences2 = retina_analysis.get_feature_sequences(RetinaFeatureAnalysis.THICKNESS_FEATURE, 7)
    retina_analysis.find_correlation(sequences1, sequences2)

    # concatenate features
    sequences = []
    for i in range(len(sequences1)):
        sequence = []
        for j in range(len(sequences1[i])):
            visit = [sequences1[i][j], sequences2[i][j]]
            sequence.append(visit)
        sequences.append(sequence)

    # create X and Y lists
    dataX = []
    dataY = []
    for i in range(len(sequences)):
        dataX_aux, dataY_aux = create_sequences(sequences[i], 2)
        dataX.append(dataX_aux)
        dataY.append(dataY_aux)

    best_model = None
    min_loss = 1000
    predictions = []
    predictionY = []
    for i in range(len(dataX)):
        if len(dataX[i]) > 4:
            lstm = Lstm(2, len(dataX[i]), dataX[i], dataY[i])
            lstm.train()

            loss, prediction = lstm.evaluate_model()
            if loss < min_loss:
                best_model = lstm
                min_loss = loss
            if prediction is not None:
                predictions.append(prediction[0])
                predictionY.append(dataY[i][len(dataY[i])-1])

    best_model.model.save('../models/lstm.h5')
    print("----------Best model----------")
    best_loss, best_output = best_model.evaluate_model()

    predictions = np.array(predictions)
    predictionY = np.array(predictionY)
    clf = linear_model.LinearRegression(fit_intercept=False)
    clf.fit(predictions, predictionY)
    print("----------Linear Regression----------")
    print("prediction: ", clf.predict(predictions))
    print("actual: ", predictionY)
    print(clf.score(predictions, predictionY))

    #labels = dtw_kmeans_clustering(sequences)
    #plot_sequences(sequences, labels)
