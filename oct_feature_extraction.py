import os
from datetime import datetime
from xml.dom import minidom

import config
import pandas as pd


class OCTFeatureExtractor:
    CENTRAL_AVG_THICKNESS_FEATURE = "CentralThickness"
    THICKNESS_FEATURE = "AvgThickness"
    VOLUME_FEATURE = "Volume"
    MIN_CENTRAL_THICKNESS_FEATURE = "MinCentralThickness"
    MAX_CENTRAL_THICKNESS_FEATURE = "MaxCentralThickness"
    TOTAL_VOLUME_FEATURE = "TotalVolume"

    def get_feature_sequences(self, feature_name, zone=0):
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
