from sklearn.preprocessing import MinMaxScaler
from trendypy.trendy import Trendy
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.utils import to_time_series_dataset
import matplotlib.pyplot as plt
import pandas as pd

from data_handling.db_handler import DbHandler


class TimeSeriesClustering:

    def __init__(self, data):
        self.data = data
        self.sequences = self.data.groupby('ID').apply(pd.DataFrame.to_numpy).to_numpy().tolist()
        self.sequences = [ts for ts in self.sequences if ts.shape[0] > 1]

    def dtw_kmeans_clustering(self, nb_clusters=2):
        print("Kmeans with DTW clustering ...")
        sequences = to_time_series_dataset(self.sequences)
        km = TimeSeriesKMeans(n_clusters=nb_clusters, metric="dtw")
        labels = km.fit_predict(sequences)
        self.plot_clusters(km, labels)
        self.compute_accuracy(labels)

    def dtw_clustering(self, nb_clusters=2):
        print("Trendypy DTW clustering ...")
        trendy = Trendy(n_clusters=nb_clusters)
        trendy.fit(self.sequences)
        labels = trendy.labels_

        self.compute_accuracy(labels)

    def kernel_kmeans_clustering(self, nb_clusters=2):
        print("Kmeans with GAK clustering ...")
        sequences = to_time_series_dataset(self.sequences)
        gak_km = KernelKMeans(n_clusters=nb_clusters, kernel="gak")
        labels = gak_km.fit_predict(sequences)

        self.compute_accuracy(labels)

    def get_actual_labels(self):
        actual_labels = []
        for nb, group in self.data.groupby('ID'):
            if group.index.size != 1:
                actual_label = 0
                if group['VA'].iloc[0] >= group['VA'].iloc[group.shape[0] - 1]:
                    actual_label = 1
                actual_labels.append(actual_label)

        return actual_labels

    def compute_accuracy(self, labels):
        accuracy = 0
        actual_labels = self.get_actual_labels()
        nb_series = len(actual_labels)
        for i in range(nb_series):
            obtained_label = labels[i]
            if obtained_label == actual_labels[i]:
                accuracy += 1

        accuracy = float(accuracy) / nb_series
        if 1 - accuracy > accuracy:
            accuracy = 1-accuracy
        print("Accuracy: " + str(accuracy))

    @staticmethod
    def plot_clusters_distribution(self, labels):
        # visualize cluster distribution
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

    # source: https://tslearn.readthedocs.io/en/stable/auto_examples/clustering/plot_kmeans.html#sphx-glr-auto-examples-clustering-plot-kmeans-py
    def plot_clusters(self, km, labels):
        size = 16
        resampled_ts = TimeSeriesResampler(sz=size).fit_transform(self.sequences)
        for yi in range(2):
            plt.subplot(3, 3, 4 + yi)
            for xx in resampled_ts[labels == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.plot(km.cluster_centers_[yi].ravel(), "r-")
            plt.xlim(0, size)
            plt.ylim(0, 4)
            plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                     transform=plt.gca().transAxes)
        plt.show()


if __name__ == '__main__':
    datatype = 'all'
    include_timestamps = True

    db_handler = DbHandler(datatype, include_timestamps)
    data = db_handler.get_data_from_csv()

    ts_clustering = TimeSeriesClustering(data)

    ts_clustering.dtw_kmeans_clustering()
    ts_clustering.kernel_kmeans_clustering()
    ts_clustering.dtw_clustering()
