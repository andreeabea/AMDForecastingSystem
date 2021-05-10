import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.svm import TimeSeriesSVC
from tslearn.utils import to_time_series_dataset

#from clustering import get_actual_labels
from regression import generate_timeseries


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


def gradient_boosted_classifier(data, include_timestamp=False, previous_visits=1, features='exclude VA'):
    X, Y = generate_timeseries(data, include_timestamp, previous_visits, features)
    gbr = GradientBoostingClassifier()

    va_set = list(set(list(Y)))
    print(va_set)
    for i in range(len(Y)):
        for j in range(len(va_set)):
            if Y[i] == va_set[j]:
                Y[i] = j

    cv = KFold(n_splits=10)
    n_scores = cross_val_score(gbr, X, Y, cv=cv, n_jobs=-1)
    print('R^2: ' + str(np.mean(n_scores)))
    #fit = gbr.fit(X, Y)
    #plot_feature_importances(fit, X.shape[1])


if __name__ == '__main__':
    #DatasetBuilder.write_all_data_to_csv("image_data.csv", datatype='images', include_timestamps=True)
    include_timestamps = True
    datatype = 'all'
    if include_timestamps:
        last_column = 'Timestamp'
        if datatype == 'numerical':
            data = pd.read_csv("all_data.csv", index_col=['ID', 'Date'], parse_dates=True)
        else:
            if datatype == 'images':
                data = pd.read_csv("image_data.csv", index_col=['ID', 'Date'], parse_dates=True)
            else:
                num_data = pd.read_csv("all_data.csv", index_col=['ID', 'Date'], parse_dates=True)
                img_data = pd.read_csv("image_data.csv", index_col=['ID', 'Date'], parse_dates=True)
                del num_data['Timestamp']
                del img_data['VA']
                del img_data['Treatment']
                data = num_data.merge(img_data, how='left', on=['ID', 'Date'])
    else:
        last_column = 'TotalVolume0'
        data = pd.read_csv("all_data_resampled.csv", index_col=['ID', 'Date'], parse_dates=True)


    data[data.columns] = MinMaxScaler().fit_transform(data)

    gradient_boosted_classifier(data, include_timestamps, 1, 'all')
