import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.svm import TimeSeriesSVC
from tslearn.utils import to_time_series_dataset

from data_handling.db_handler import DbHandler
from data_handling.timeseries_augmentation import TimeSeriesGenerator


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
            if group['VA'].iloc[0] >= group['VA'].iloc[group.shape[0] - 1]:
                actual_label = 1
            actual_labels.append(actual_label)

        return actual_labels

    def knn_classifier(self, nb_neighbors):
        knn = KNeighborsTimeSeriesClassifier(n_neighbors=nb_neighbors, metric="dtw")
        trainX, trainY, testX, testY = self.split_data()
        print(knn.fit(trainX, trainY).score(testX, testY))

    def svc_classifier(self):
        svc = TimeSeriesSVC(kernel="gak", gamma="auto", probability=True)
        trainX, trainY, testX, testY = self.split_data()
        print(svc.fit(trainX, trainY).score(testX, testY))

    def gradient_boosted_classifier(self, include_timestamp=False, previous_visits=1, features='exclude VA'):
        print("Gradient boosting classifier ...")
        X, Y = self.gen.generate_timeseries(include_timestamp, previous_visits, features)
        gbr = GradientBoostingClassifier()

        va_set = list(set(list(Y)))
        for i in range(len(Y)):
            for j in range(len(va_set)):
                if Y[i] == va_set[j]:
                    Y[i] = j

        cv = KFold(n_splits=10)
        n_scores = cross_val_score(gbr, X, Y, cv=cv, n_jobs=-1)
        print('R^2: ' + str(np.mean(n_scores)))


if __name__ == '__main__':
    #DatasetBuilder.write_all_data_to_csv("image_data.csv", datatype='images', include_timestamps=True)
    include_timestamps = True
    datatype = 'numerical'

    db_handler = DbHandler(datatype, include_timestamps)
    data = db_handler.get_data_from_csv()

    ts_classifier = TimeSeriesClassifier(data)
    ts_classifier.gradient_boosted_classifier(include_timestamps, 1, 'all')
