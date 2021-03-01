import os
from xml.dom import minidom

from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from sklearn.preprocessing import StandardScaler
from trendypy.trendy import Trendy
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.utils import to_time_series_dataset

import config
from data_layer.build_dataset import read_all_images, reshape_images
from code_handler import CodeHandler


class ImageAnalysis:

    def __init__(self):
        self.code_handler = CodeHandler()
        self.image_width = self.code_handler.image_width
        self.image_height = self.code_handler.image_height

    def reshape_images_pca(self, images):
        images = np.vstack(images)
        images = images.reshape(-1, self.image_height, self.image_width, 1)
        print(images.shape)
        images = images.reshape(images.shape[0], self.image_width * self.image_height)
        print("Reshaped...")

        return images

    # dimensionality reduction methods for the images, as well for the latent codes
    def pca2(self, images):
        pca = PCA(2)  # project to 2 dimensions
        projected = pca.fit_transform(images)

        plt.scatter(projected[:, 0], projected[:, 1],
                    edgecolor='none', alpha=0.5,
                    color='blue')
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.savefig("plot-pca2.png")

    def tSNE(self, data, datatype):
        # first reduce dimensionality - if the given data are images
        if datatype == 0:
            pca = PCA(256)
            data = pca.fit_transform(data)

        # reduce dimensionality with t-SNE
        tsne = TSNE(n_components=2, verbose=1, perplexity=256, n_iter=1000, learning_rate=0.1)
        tsne_results = tsne.fit_transform(data)

        # visualize
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
                    edgecolor='none', alpha=0.5,
                    color='blue')
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.savefig("plots/t-SNE2.png")

    def find_cum_explained_variance(self, images):
        # find the PCA dimensionality to represent most of the data
        sc = StandardScaler()
        images = sc.fit_transform(images)

        print("computing pca")
        # find the number of dimensions to represent at least 80% of data
        pca = PCA(n_components=0.8, svd_solver='full').fit(images)
        print(pca.n_components_)
        print("obtained PCA\nPlotting...")
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        print("Saving...")
        plt.savefig("plots/cumulative-explained-variance.png")

    def get_latent_code_sequences(self):
        sequences = []

        for patient in os.scandir(config.ORIG_INPUT_DATASET):
            visits = len(list(os.scandir(patient)))
            if (visits <= 1):
                continue
            left_eye_images = []
            right_eye_images = []

            for visit in os.scandir(patient):
                for eye in os.scandir(visit):
                    fundusImgPath = ""
                    for file in os.scandir(eye):
                        if file.path.endswith(".xml"):
                            xmlfile = minidom.parse(file.path)
                            fundusImgPath = xmlfile.getElementsByTagName('ExamURL')[0].firstChild.nodeValue.rsplit("\\",1)[1]
                    for imageFile in os.scandir(eye):
                        if imageFile.path.endswith(".tif") and fundusImgPath not in imageFile.name:
                            image = Image.open(imageFile.path).convert("L")
                            # resize img
                            image = np.array(image.resize((self.image_width, self.image_height), Image.ANTIALIAS))

                            if image.shape[0] == self.image_width and image.shape[1] == self.image_height:
                                image = image.reshape(self.image_width, self.image_height, 1)
                                if "OS" in eye.name:
                                    left_eye_images.append(image)
                                else:
                                    right_eye_images.append(image)
                            break

            if len(left_eye_images) > 1:
                left_eye_images = reshape_images(left_eye_images, self.image_width, self.image_height)
                left_eye_images = left_eye_images / 255
                latent_code_sequences = self.code_handler.get_latent_codes(left_eye_images)
                sequences.append(latent_code_sequences.tolist())
            if len(right_eye_images) > 1:
                right_eye_images = reshape_images(right_eye_images, self.image_width, self.image_height)
                right_eye_images = right_eye_images / 255
                latent_code_sequences = self.code_handler.get_latent_codes(right_eye_images)
                sequences.append(latent_code_sequences.tolist())

        return sequences

    def plot_sequence_clusters(self, sequences, labels):
        pca = PCA(2)  # project to 2 dimensions
        for i in range(len(sequences)):
            projected = pca.fit_transform(sequences[i])

            cmap = ["red", "blue"]
            color = cmap[labels[i]]
            plt.scatter(projected[:, 0], projected[:, 1], alpha=0.5, color=color)
            plt.xlabel('component 1')
            plt.ylabel('component 2')
            plt.savefig("plots/sequence-clusters-pca.png")

    # Dynamic Time Warping Clustering using TrendyPy library
    def dtw_clustering(self, sequences, nb_clusters):
        trendy = Trendy(n_clusters=nb_clusters)
        trendy.fit(sequences)
        labels = trendy.labels_

        return labels

    # Kmeans clustering using DTW distance instead of Euclidean distance
    def kmeans_dtw_clustering(self, sequences, nb_clusters):
        sz = 16
        sequences = TimeSeriesResampler(sz=sz).fit_transform(sequences)
        sequences = to_time_series_dataset(sequences)
        km = TimeSeriesKMeans(n_clusters=nb_clusters, metric="softdtw")
        labels = km.fit_predict(sequences)

        # TODO: add code source
        for yi in range(nb_clusters):
            plt.subplot(3, 3, 4 + yi)
            for xx in sequences[labels == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.plot(km.cluster_centers_[yi].ravel(), "r-")
            plt.xlim(0, sz)
            plt.ylim(0, 4)
            plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                     transform=plt.gca().transAxes)
            if yi == 1:
                plt.title("DBA $k$-means")

        plt.show()

        return labels

    def get_labels_dictionary(self):
        keys = []
        code_sequences = self.get_latent_code_sequences()
        labels = self.kmeans_dtw_clustering(code_sequences)

        for patient in os.scandir(config.ORIG_INPUT_DATASET):
            visits = len(list(os.scandir(patient)))
            if (visits <= 1):
                continue
            for visit in os.scandir(patient):
                for eye in os.scandir(visit):
                    for file in os.scandir(eye):
                        if file.path.endswith(".xml"):
                            xmlfile = minidom.parse(file.path)
                            patientID = xmlfile.getElementsByTagName('ID')[0].firstChild.nodeValue

                    if "OS" in eye.name:
                        key = str(patientID) + "OS"
                        keys.append(key)
                    else:
                        if "5018" not in patientID:
                            key = str(patientID) + "OD"
                            keys.append(key)
                break

        return dict(zip(keys, labels))

    def dimensionality_reduction(self):
        # 2 approaches: apply dimensionality reduction on images or on codes obtained from the autoencoder's latent space
        x, y, images = read_all_images()
        images = self.reshape_images_pca(images)
        images = images/255
        #latent_codes = get_latent_codes(images)
        #pca2(images)
        self.tSNE(images, 0)

        self.find_cum_explained_variance(images)


if __name__ == '__main__':
    img_analysis = ImageAnalysis()

    code_sequences = img_analysis.get_latent_code_sequences()
    labels = img_analysis.dtw_clustering(code_sequences)
    #img_analysis.plot_sequence_clusters(code_sequences, labels)

    #img_analysis.dimensionality_reduction()
