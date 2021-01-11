import os
from xml.dom import minidom

from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, Isomap
import numpy as np
from sklearn.preprocessing import StandardScaler
#from trendypy.trendy import Trendy
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

import config
from build_dataset import read_all_images, split_data, reshape_images
from prediction import get_latent_codes


def reshape_images_pca(images, x, y):
    images = np.vstack(images)
    images = images.reshape(-1, x, y, 1)
    print(images.shape)
    images = images.reshape(images.shape[0], x * y)
    print("Reshaped...")
    return images


def pca2(images):
    pca = PCA(2)  # project to 2 dimensions
    projected = pca.fit_transform(images)
    #print(images.shape)
    #print(projected.shape)

    plt.scatter(projected[:, 0], projected[:, 1],
                edgecolor='none', alpha=0.5,
                color='blue')
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.savefig("plotC0.png")


def tSNE(data, datatype):
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


def find_cum_explained_variance(images):
    # find the PCA dimensionality to represent most of the data (99%)
    sc = StandardScaler()
    images = sc.fit_transform(images)

    print("computing pca")
    pca = PCA(n_components=0.8, svd_solver='full').fit(images)
    print(pca.n_components_)
    print("obtained PCA\nPlotting...")
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    print("Saving...")
    plt.savefig("plots/cumulative-explained-variance.png")


def get_latent_code_sequences():
    x = 256
    y = 256
    sequences = []

    for patient in os.scandir(config.ORIG_INPUT_DATASET):
        visits = len(list(os.scandir(patient)))
        #print(patient.name + " " + str(visits))
        if (visits <= 1):
            continue
        left_eye_images = []
        right_eye_images = []
        for visit in os.scandir(patient):
            #print(visit.name)
            for eye in os.scandir(visit):
                fundusImgPath = ""
                for file in os.scandir(eye):
                    if file.path.endswith(".xml"):
                        xmlfile = minidom.parse(file.path)
                        fundusImgPath = xmlfile.getElementsByTagName('ExamURL')[0].firstChild.nodeValue.rsplit("\\", 1)[1]
                for imageFile in os.scandir(eye):
                    if imageFile.path.endswith(".tif") and fundusImgPath not in imageFile.name:
                        image = Image.open(imageFile.path).convert("L")
                        # resize img
                        image = np.array(image.resize((256, 256), Image.ANTIALIAS))

                        if image.shape[0] == x and image.shape[1] == y:
                            image = image.reshape(x, y, 1)
                            if "OS" in eye.name:
                                left_eye_images.append(image)
                            else:
                                right_eye_images.append(image)
                        break

        #if len(left_eye_images) == 1 or len(right_eye_images) == 1:
        #    print(patient.name)

        if len(left_eye_images) > 1:
            left_eye_images = reshape_images(left_eye_images, x, y)
            left_eye_images = left_eye_images / 255
            latent_code_sequences = get_latent_codes(left_eye_images)
            sequences.append(latent_code_sequences.tolist())
        if len(right_eye_images) > 1:
            right_eye_images = reshape_images(right_eye_images, x, y)
            right_eye_images = right_eye_images / 255
            latent_code_sequences = get_latent_codes(right_eye_images)
            sequences.append(latent_code_sequences.tolist())

    return sequences


def plot_sequence_clusters(sequences, labels):
    pca = PCA(2)  # project to 2 dimensions
    for i in range(len(sequences)):
        projected = pca.fit_transform(sequences[i])

        cmap = ["red", "blue"]
        color = cmap[labels[i]]
        plt.scatter(projected[:, 0], projected[:, 1], alpha=0.5, color=color)
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.savefig("plots/pca-labelled256-kmeans2.png")


def dtw_clustering(sequences):
    #trendy = Trendy(n_clusters=2)
    #trendy.fit(code_sequences)
    #labels = trendy.labels_
    sequences = to_time_series_dataset(sequences)
    km = TimeSeriesKMeans(n_clusters=2, metric="softdtw")
    labels = km.fit_predict(sequences)
    return labels


def get_labels_dictionary():
    keys = []
    code_sequences = get_latent_code_sequences()
    labels = dtw_clustering(code_sequences)

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


def isomap(data):
    model = Isomap(n_components=2)
    proj = model.fit_transform(data)

    ax = plt.gca()

    proj = model.fit_transform(data)
    ax.plot(proj[:, 0], proj[:, 1], '.k')


def dimensionality_reduction():
    # 2 approaches: apply dimensionality reduction on images or on codes obtained from the autoencoder's latent space
    x, y, images = read_all_images()
    images = reshape_images_pca(images, x, y)
    images = images/255
    #latent_codes = get_latent_codes(images)
    #pca2(images)
    tSNE(images, 0)

    #find_cum_explained_variance(images)


#code_sequences = get_latent_code_sequences()
#labels = dtw_clustering(code_sequences)
#plot_sequence_clusters(code_sequences, labels)

#dimensionality_reduction()
