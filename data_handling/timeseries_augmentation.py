import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


class TimeSeriesGenerator:

    def __init__(self, data):
        self.data = data

    # overlapping sliding window time-series segmentation
    def sliding_window(self, size):
        X = []
        Y = []

        sequences = self.data.groupby('ID').apply(pd.DataFrame.to_numpy).to_numpy().tolist()
        for i in range(len(sequences)):
            if len(sequences[i]) > size:
                generator = TimeseriesGenerator(sequences[i], sequences[i], length=size)
                x, y = generator[0]
                X.append(x)
                Y.append(y)

        return X, Y

    def generate_timeseries(self, include_timestamp=False, size=1, features='all'):
        X, Y = self.sliding_window(size)

        X = np.vstack(X)
        Y = np.vstack(Y)

        # select most important features: 45% acc?
        feature_list = [22, 23, 17, 21, 11, 12, 7, 20, 16]

        if include_timestamp:
            # update the timestamps
            for i in range(len(X)):
                min_timestamp = X[i, 0, X.shape[2] - 1]
                X[i, :, X.shape[2] - 1] = np.subtract(X[i, :, X.shape[2] - 1], min_timestamp)
                Y[i, Y.shape[1] - 1] = Y[i, Y.shape[1] - 1] - min_timestamp
            # split target visual acuity and move the future timestamp to X
            future_timestamp_array = Y[:, Y.shape[1] - 1:]
            # append the timestamp to the feature list
            feature_list.append(24)
        Y = Y[:, :1]
        if features == 'exclude VA':
            X = X[:, :, 1:]
        else:
            if features != 'all':
                if features == 'default':
                    X = X[:, :, feature_list]
                else:
                    if 25 in features:
                        X = X[:, :, features[:-1]]
                    else:
                        X = X[:, :, features]
        X = X.reshape(-1, X.shape[1] * X.shape[2])
        Y = Y.reshape(-1)
        if include_timestamp and future_timestamp_array.shape[1] != 0:
            X = np.hstack((X, future_timestamp_array))

        # pca = PCA(20)
        # X = pca.fit_transform(X)
        # tsne = TSNE(n_components=3, verbose=1, perplexity=25, n_iter=1000, learning_rate=0.1)
        # X = tsne.fit_transform(X)
        return X, Y