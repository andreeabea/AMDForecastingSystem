import os
from datetime import datetime
from xml.dom import minidom

from PIL import Image

import config
import pandas as pd
import numpy as np

from data_processing.latent_code_handler import LatentCodeHandler


class OCTFeatureExtractor:
    CENTRAL_AVG_THICKNESS_FEATURE = "CentralThickness"
    THICKNESS_FEATURE = "AvgThickness"
    VOLUME_FEATURE = "Volume"
    MIN_CENTRAL_THICKNESS_FEATURE = "MinCentralThickness"
    MAX_CENTRAL_THICKNESS_FEATURE = "MaxCentralThickness"
    TOTAL_VOLUME_FEATURE = "TotalVolume"

    feature_names = [THICKNESS_FEATURE,
                     VOLUME_FEATURE,
                     CENTRAL_AVG_THICKNESS_FEATURE,
                     MIN_CENTRAL_THICKNESS_FEATURE,
                     MAX_CENTRAL_THICKNESS_FEATURE,
                     TOTAL_VOLUME_FEATURE]

    def get_feature_df(self, feature_name, zone=0):
        # method to get features from the XML files
        # retina zone parameter ranges from 0 to 8 : C0, N1, N2, S1, S2, T1, T2, I1, I2
        df = pd.DataFrame()

        for patient in os.scandir(config.ORIG_INPUT_DATASET):
            visits = len(list(os.scandir(patient)))

            if (visits <= 1):
                continue

            for visit in os.scandir(patient):
                for eye in os.scandir(visit):
                    date = datetime.strptime(visit.name.split("- ")[1], '%d.%m.%Y')
                    for file in os.scandir(eye):
                        if file.path.endswith(".xml"):
                            xmlfile = minidom.parse(file.path)
                            feature = xmlfile.getElementsByTagName(feature_name)
                            if feature is not None and feature[zone].firstChild is not None:
                                patientID = xmlfile.getElementsByTagName('ID')[0].firstChild.nodeValue
                                column_name = feature_name + str(zone)
                                if "OS" in eye.name:
                                    newEntryL = pd.DataFrame({'ID': [str(patientID) + 'OS'],
                                                              'Date': [date],
                                                              column_name: [float(feature[zone].firstChild.nodeValue)]})
                                    df = df.append(newEntryL)
                                else:
                                    newEntryR = pd.DataFrame({'ID': [str(patientID) + 'OD'],
                                                              'Date': [date],
                                                              column_name: [float(feature[zone].firstChild.nodeValue)]})
                                    df = df.append(newEntryR)

        return df

    def get_all_features(self):
        result_df = pd.DataFrame()
        df_features = []
        for i in range(len(OCTFeatureExtractor.feature_names)):
            if i <= 1:
                nb_zones = 9
            else:
                nb_zones = 1
            for j in range(nb_zones):
                df = self.get_feature_df(OCTFeatureExtractor.feature_names[i], j)
                feature_name = OCTFeatureExtractor.feature_names[i] + str(j)
                df_features.append(feature_name)
                try:
                    result_df = result_df.merge(df, how='left', on=['ID', 'Date'])
                except KeyError:
                    result_df = df.copy()
                result_df = result_df.sort_values(['ID', 'Date'], ascending=[True, True])

        return result_df, df_features

    def get_images_df(self):
        df = pd.DataFrame()
        code_handler = LatentCodeHandler('../models/autoencoder256-best.h5')

        for patient in os.scandir(config.ORIG_INPUT_DATASET):
            visits = len(list(os.scandir(patient)))

            if (visits <= 1):
                continue

            for visit in os.scandir(patient):
                for eye in os.scandir(visit):
                    date = datetime.strptime(visit.name.split("- ")[1], '%d.%m.%Y')
                    for file in os.scandir(eye):
                        if file.path.endswith(".xml"):
                            xmlfile = minidom.parse(file.path)
                            fundusImgPath = ""
                            for file in os.scandir(eye):
                                if file.path.endswith(".xml"):
                                    xmlfile = minidom.parse(file.path)
                                    fundusImgPath = \
                                    xmlfile.getElementsByTagName('ExamURL')[0].firstChild.nodeValue.rsplit("\\", 1)[1]
                    for imageFile in os.scandir(eye):
                        if imageFile.path.endswith(".tif") and fundusImgPath is not None and fundusImgPath in imageFile.name:
                            image = Image.open(imageFile.path).convert("L")
                            # resize img
                            image = np.array(image.resize((256, 256), Image.ANTIALIAS))
                            patientID = xmlfile.getElementsByTagName('ID')[0].firstChild.nodeValue
                            image = self.reshape_image(np.array(image), 256, 256)
                            image = image / 255
                            latent_code = code_handler.get_latent_codes(image)
                            codes_dict = {}
                            for i in range(len(latent_code[0])):
                                codes_dict[str(i)] = latent_code[0][i]

                            if "OS" in eye.name:
                                newEntryL = pd.DataFrame({**dict({'ID': [str(patientID) + 'OS'], 'Date': [date]}), **codes_dict})
                                df = df.append(newEntryL)
                            else:
                                newEntryR = pd.DataFrame({**dict({'ID': [str(patientID) + 'OD'], 'Date': [date]}), **codes_dict})
                                df = df.append(newEntryR)

        return df

    def reshape_image(self, images, x, y):
        images = np.vstack(images)
        images = images.reshape(-1, x, y, 1)
        return images

    @staticmethod
    def get_fundus_paths_df():
        df = pd.DataFrame()

        for patient in os.scandir(config.ORIG_INPUT_DATASET):
            visits = len(list(os.scandir(patient)))

            if (visits <= 1):
                continue

            for visit in os.scandir(patient):
                for eye in os.scandir(visit):
                    date = datetime.strptime(visit.name.split("- ")[1], '%d.%m.%Y')
                    for file in os.scandir(eye):
                        if file.path.endswith(".xml"):
                            xmlfile = minidom.parse(file.path)
                            fundusImgPath = xmlfile.getElementsByTagName('ExamURL')[0].firstChild.nodeValue.rsplit("\\", 1)[1]
                            patientID = xmlfile.getElementsByTagName('ID')[0].firstChild.nodeValue
                    fundusImgPath = './data/' + patient.name + '/' + visit.name + '/' + eye.name + '/' + fundusImgPath
                    if "OS" in eye.name:
                        newEntryL = pd.DataFrame({'ID': [str(patientID) + 'OS'], 'Date': [date], 'path': fundusImgPath})
                        df = df.append(newEntryL)
                    else:
                        newEntryR = pd.DataFrame({'ID': [str(patientID) + 'OD'], 'Date': [date], 'path': fundusImgPath})
                        df = df.append(newEntryR)

        return df


if __name__ == '__main__':
    OCTFeatureExtractor.get_fundus_paths_df().to_csv('../preprocessed_data/fundus_img_paths.csv')
