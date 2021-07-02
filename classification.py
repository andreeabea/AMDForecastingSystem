import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.svm import TimeSeriesSVC
from tslearn.utils import to_time_series_dataset

from data_processing.db_handler import DbHandler
from data_processing.timeseries_augmentation import TimeSeriesGenerator
from neural_networks.cnn import Cnn
from regression import TimeSeriesRegressor
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class TimeSeriesClassifier:

    def __init__(self, data):
        self.data = data
        self.gen = TimeSeriesGenerator(data)

    def split_data(self):
        labels = np.array(self.get_actual_labels())
        mask = np.random.rand(len(labels)) < 0.8
        trainY = labels[mask]
        testY = labels[~mask]

        i = 0
        trainX = []
        testX = []
        for nb, group in self.data.groupby('ID'):
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

    def get_actual_labels(self):
        actual_labels = []
        for nb, group in self.data.groupby('ID'):
            actual_label = 0
            if group['VA'].iloc[0] > group['VA'].iloc[group.shape[0] - 1]:
                actual_label = 1
            actual_labels.append(actual_label)

        return actual_labels

    def knn_classifier(self, nb_neighbors):
        knn = KNeighborsTimeSeriesClassifier(n_neighbors=nb_neighbors, metric="dtw")
        trainX, trainY, testX, testY = self.split_data()
        knn = knn.fit(trainX, trainY)
        print(knn.score(testX, testY))
        conf_matrix = confusion_matrix(testY, knn.predict(testX))
        sns.heatmap(conf_matrix, annot=True)
        plt.show()

    def svc_classifier(self):
        print("Support vector classifier ...")
        svc = TimeSeriesSVC(kernel="gak", gamma="auto", probability=True)
        trainX, trainY, testX, testY = self.split_data()
        print(svc.fit(trainX, trainY).score(testX, testY))

    def gradient_boosted_classifier(self, include_timestamp=False, previous_visits=1, features='exclude VA'):
        print("Gradient boosting classifier ...")
        X, Y = self.gen.generate_timeseries(include_timestamp, previous_visits, features)
        XwithVA, _ = self.gen.generate_timeseries(include_timestamp, previous_visits, 'all')
        gbr = GradientBoostingClassifier()

        # get VA distinct labels
        # va_set = list(set(list(Y)))
        # for i in range(len(Y)):
        #     for j in range(len(va_set)):
        #         if Y[i] == va_set[j]:
        #             Y[i] = j

        # get VA distinct labels: good/bad evolution
        for i in range(len(Y)):
            if Y[i] > XwithVA[i][0]:
                Y[i] = -1
            else:
                Y[i] = 1

        cv = KFold(n_splits=10)
        n_scores = cross_val_score(gbr, X, Y, cv=cv, n_jobs=-1)
        print('Accuracy: ' + str(np.mean(n_scores)))
        pred = cross_val_predict(gbr, X, Y, cv=cv, n_jobs=-1)
        conf_matrix = confusion_matrix(Y, pred, labels=[-1, 1])
        sns.heatmap(conf_matrix, annot=True, yticklabels=['Actual good evolution', 'Actual bad evolution'])
        plt.show()

    def cnn_classifier(self, include_timestamp=False, previous_visits=1, features='exclude VA'):
        X, Y = self.gen.generate_timeseries(include_timestamp, previous_visits, features)
        XwithVA, _ = self.gen.generate_timeseries(include_timestamp, previous_visits, 'all')
        # get VA distinct labels: good/bad evolution
        newY = np.array([])
        for i in range(len(Y)):
            if Y[i] > XwithVA[i][0]:
                newY = np.append(newY, np.array([1, 0]))
            else:
                newY = np.append(newY, np.array([0, 1]))

        newY = newY.reshape(-1, 2)

        trainX, trainY, validX, validY, testX, testY = TimeSeriesRegressor.train_test_val_split(X, newY)
        cnn = Cnn(trainX, trainY, validX, validY, testX, testY, nb_labels=2)
        cnn.train()

        cnn.evaluate_model()


if __name__ == '__main__':
    #DatasetBuilder.write_all_data_to_csv("image_data.csv", datatype='images', include_timestamps=True)
    include_timestamps = True
    datatype = 'all'

    db_handler = DbHandler(datatype, include_timestamps)
    data = db_handler.get_data_from_csv()

    ts_classifier = TimeSeriesClassifier(data)
    ts_classifier.gradient_boosted_classifier(include_timestamps, 1, 'exclude VA')

    ts_classifier.knn_classifier(2)
    #ts_classifier.cnn_classifier(include_timestamps, 3, 'all')
