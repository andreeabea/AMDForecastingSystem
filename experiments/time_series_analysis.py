import math
import os
from datetime import datetime
from xml.dom import minidom

from keras_preprocessing.sequence import TimeseriesGenerator
from trendypy.trendy import Trendy
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.utils import to_time_series_dataset

import config
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from data_layer.build_dataset import create_sequences, find_visual_acuity, append_va_to_features
from neural_networks.lstm import Lstm

from visual_acuity_analysis import VisualAcuityAnalysis


class RetinaFeatureAnalysis:

    CENTRAL_AVG_THICKNESS_FEATURE = "CentralThickness"
    THICKNESS_FEATURE = "AvgThickness"
    VOLUME_FEATURE = "Volume"
    MIN_CENTRAL_THICKNESS_FEATURE = "MinCentralThickness"
    MAX_CENTRAL_THICKNESS_FEATURE = "MaxCentralThickness"
    TOTAL_VOLUME_FEATURE = "TotalVolume"

    def get_feature_sequences(self, feature_name, zone=0, eyeData=None, include_va=False):
        sequences = []

        for patient in os.scandir(config.ORIG_INPUT_DATASET):
            visits = len(list(os.scandir(patient)))

            if (visits <= 1):
                continue

            featurelist_left = []
            featurelist_right = []
            left_eye_va = []
            right_eye_va = []

            for visit in os.scandir(patient):
                for eye in os.scandir(visit):
                    date = datetime.strptime(visit.name.split("- ")[1], '%d.%m.%Y')
                    for file in os.scandir(eye):
                        if file.path.endswith(".xml"):
                            xmlfile = minidom.parse(file.path)
                            patientID = xmlfile.getElementsByTagName('ID')[0].firstChild.nodeValue
                            feature = xmlfile.getElementsByTagName(feature_name)
                            if feature is not None and feature[zone].firstChild is not None:
                                if "OS" in eye.name:
                                    if eyeData is not None:
                                        visual_acuity = find_visual_acuity(eyeData, patientID, "OS", date)
                                        if visual_acuity >= 0:
                                            left_eye_va.append(visual_acuity)
                                            featurelist_left.append(float(feature[zone].firstChild.nodeValue))
                                    else:
                                        featurelist_left.append(float(feature[zone].firstChild.nodeValue))
                                else:
                                    if eyeData is not None:
                                        visual_acuity = find_visual_acuity(eyeData, patientID, "OD", date)
                                        if visual_acuity >= 0:
                                            right_eye_va.append(visual_acuity)
                                            featurelist_right.append(float(feature[zone].firstChild.nodeValue))
                                    else:
                                        featurelist_right.append(float(feature[zone].firstChild.nodeValue))

            if len(featurelist_left) > 1:
                if eyeData is not None and include_va is True:
                    featurelist_left = append_va_to_features(featurelist_left, left_eye_va)
                sequences.append(featurelist_left)
            if len(featurelist_right) > 1:
                if eyeData is not None and include_va is True:
                    featurelist_right = append_va_to_features(featurelist_right, right_eye_va)
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
        sz = 16
        #sequences = TimeSeriesResampler(sz=sz).fit_transform(sequences)
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

    def concatenate_features(self, sequences1, sequences2):
        if len(sequences1) == 0:
            return sequences2
        else:
            if len(sequences2) == 0:
                return sequences1

        sequences = []

        for i in range(min(len(sequences1), len(sequences2))):
            sequence = []
            if len(sequences1[i]) == len(sequences2[i]):
                for j in range(len(sequences1[i])):
                    if isinstance(sequences1[i][j], np.ndarray):
                        visit = np.append(sequences1[i][j], sequences2[i][j])
                    else:
                        if isinstance(sequences1[i][j], list):
                            if isinstance(sequences2[i][j], np.ndarray):
                                visit = sequences1[i][j]
                                for k in range(len(sequences2[i][j])):
                                    visit = np.append(visit, sequences2[i][j][k])
                            else:
                                visit = sequences1[i][j].append(sequences2[i][j])
                        else:
                            if isinstance(sequences2[i][j], np.ndarray):
                                visit = np.append(sequences2[i][j][0:len(sequences2[i][j])-1], sequences1[i][j])
                                visit = np.append(visit, sequences2[i][j][-1])
                            else:
                                visit = [sequences1[i][j], sequences2[i][j]]
                    sequence.append(visit)
                sequences.append(sequence)

        return sequences

    def create_XY(self, sequences, nb_features):
        dataX = []
        dataY = []
        for i in range(len(sequences)):
            dataX_aux, dataY_aux = create_sequences(sequences[i], nb_features)
            dataX.append(dataX_aux)
            dataY.append(dataY_aux)

        return dataX, dataY

    def dtw_clustering(self, sequences):
        trendy = Trendy(n_clusters=2)
        sequences = TimeSeriesResampler(sz=16).fit_transform(sequences)
        trendy.fit(sequences)
        labels = trendy.labels_

        return labels

    def extract_last_feature_list(self, sequences):
        va_sequences = []
        for i in range(len(sequences)):
            va_sequence = []
            for j in range(len(sequences[i])):
                # visual acuity is the last feature
                va_sequence.append(sequences[i][j][-1])
                sequences[i][j] = sequences[i][j][0:len(sequences[i][j])-1]
            va_sequences.append(va_sequence)

        return sequences, va_sequences

    def compare_va_labels(self, sequences):
        # get visual acuity list
        sequences, va_sequences = retina_analysis.extract_last_feature_list(sequences)

        labels = retina_analysis.dtw_clustering(sequences)
        actual_labels = retina_analysis.dtw_clustering(va_sequences)

        accuracy = 0
        for i in range(len(labels)):
            if labels[i] == actual_labels[i]:
                accuracy += 1

        accuracy = float(accuracy) / len(labels)
        print("accuracy: ", accuracy)

    def compute_correlations_with_va(self):
        feature_names = [RetinaFeatureAnalysis.THICKNESS_FEATURE,
                         RetinaFeatureAnalysis.VOLUME_FEATURE,
                         RetinaFeatureAnalysis.CENTRAL_AVG_THICKNESS_FEATURE,
                         RetinaFeatureAnalysis.MIN_CENTRAL_THICKNESS_FEATURE,
                         RetinaFeatureAnalysis.MAX_CENTRAL_THICKNESS_FEATURE,
                         RetinaFeatureAnalysis.TOTAL_VOLUME_FEATURE]

        for i in range(len(feature_names)):
            if i <= 1:
                nb_zones = 9
            else:
                nb_zones = 1
            for j in range(nb_zones):
                sequences1 = retina_analysis.get_feature_sequences(feature_names[i], j, eyeData)
                sequences2 = retina_analysis.get_feature_sequences(feature_names[i], j, eyeData, include_va=True)
                sequences, va_sequences = retina_analysis.extract_va_list(sequences2)
                print("VA correlation with " + feature_names[i] + " in zone " + str(j))
                retina_analysis.find_correlation(sequences1, va_sequences)
                #input()

    # TODO: not working!
    def get_all_features(self):
        feature_names = [RetinaFeatureAnalysis.THICKNESS_FEATURE,
                         RetinaFeatureAnalysis.VOLUME_FEATURE,
                         RetinaFeatureAnalysis.CENTRAL_AVG_THICKNESS_FEATURE,
                         RetinaFeatureAnalysis.MIN_CENTRAL_THICKNESS_FEATURE,
                         RetinaFeatureAnalysis.MAX_CENTRAL_THICKNESS_FEATURE,
                         RetinaFeatureAnalysis.TOTAL_VOLUME_FEATURE]

        sequences = []

        for i in range(len(feature_names) - 1):
            if i <= 1:
                nb_zones = 9
            else:
                nb_zones = 1
            for j in range(nb_zones):
                sequences1 = retina_analysis.get_feature_sequences(feature_names[i], j, eyeData)
                sequences = retina_analysis.concatenate_features(sequences, sequences1)

        sequences2 = retina_analysis.get_feature_sequences(feature_names[-1], 0, eyeData, include_va=True)
        sequences = retina_analysis.concatenate_features(sequences, sequences2)

        return sequences


if __name__ == '__main__':

    va_analysis = VisualAcuityAnalysis()
    eyeData = va_analysis.get_va_df()

    retina_analysis = RetinaFeatureAnalysis()

    sequences1 = retina_analysis.get_feature_sequences(RetinaFeatureAnalysis.THICKNESS_FEATURE, 3, eyeData)
    sequences2 = retina_analysis.get_feature_sequences(RetinaFeatureAnalysis.THICKNESS_FEATURE, 7, eyeData, True)
    # sequences, va_sequences = retina_analysis.extract_va_list(sequences2)
    #retina_analysis.find_correlation(sequences1, sequences2)

    sequences = retina_analysis.concatenate_features(sequences1, sequences2)

    sequences = retina_analysis.get_all_features()

    if isinstance(sequences[0][0], float):
        nb_features = 1
    else:
        nb_features = len(sequences[0][0])

    # create X and Y lists
    dataX, dataY = retina_analysis.create_XY(sequences, nb_features)

    lstm = Lstm(nb_features, 1, dataX, dataY)
    lstm.train()

    loss, prediction = lstm.evaluate_model()

    #labels = dtw_kmeans_clustering(sequences)
    #plot_sequences(sequences, labels)

    #retina_analysis.compute_correlations_with_va()
