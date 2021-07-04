from datetime import datetime

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DbHandler:

    def __init__(self, datatype='all', include_timestamps=True):
        self.datatype = datatype
        self.include_timestamps = include_timestamps

    def get_data_from_csv(self, normalize=True):
        if self.include_timestamps:
            if self.datatype == 'numerical':
                data = pd.read_csv("preprocessed_data/num_data.csv", index_col=['ID', 'Date'], parse_dates=True)
            else:
                if self.datatype == 'images':
                    data = pd.read_csv("preprocessed_data/image_data.csv", index_col=['ID', 'Date'], parse_dates=True)
                else:
                    data = self.get_all_csv_data()
        else:
            if self.datatype == 'numerical':
                data = pd.read_csv("preprocessed_data/num_data_resampled.csv", index_col=['ID', 'Date'],
                                   parse_dates=True)
            else:
                if self.datatype == 'images':
                    data = pd.read_csv("preprocessed_data/image_data_resampled.csv", index_col=['ID', 'Date'],
                                       parse_dates=True)
                else:
                    data = self.get_all_csv_data()

        # normalize data
        if normalize:
            data[data.columns] = MinMaxScaler().fit_transform(data)

        return data

    def get_all_csv_data(self):
        if self.include_timestamps:
            num_data = pd.read_csv("preprocessed_data/num_data.csv", index_col=['ID', 'Date'], parse_dates=True)
            img_data = pd.read_csv("preprocessed_data/image_data.csv", index_col=['ID', 'Date'], parse_dates=True)
            del num_data['Timestamp']
        else:
            num_data = pd.read_csv("preprocessed_data/num_data_resampled.csv", index_col=['ID', 'Date'],
                                   parse_dates=True)
            img_data = pd.read_csv("preprocessed_data/image_data_resampled.csv", index_col=['ID', 'Date'],
                                   parse_dates=True)

        del img_data['VA']
        del img_data['Treatment']
        data = num_data.merge(img_data, how='left', on=['ID', 'Date'])

        return data

    @staticmethod
    def get_fundus_img_path(patient_id, date):
        data = pd.read_csv("preprocessed_data/fundus_img_paths.csv", index_col=['ID', 'Date'],
                           parse_dates=True)
        path = data.loc[(data.index.get_level_values(0) == patient_id) &
                        (data.index.get_level_values(1).strftime('%Y-%m-%d') == date)]
        return path.iat[0,0]
