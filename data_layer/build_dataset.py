import re

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from classification import learning_shapelets_classifier
from clustering import dtw_kmeans_clustering, dtw_clustering, kernel_kmeans_clustering
from regression import svr_regression


class DatasetBuilder:

    def __init__(self):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.options.display.float_format = '{:,.2f}'.format

        xls = pd.ExcelFile(r"DMLVAVcuID.xls")
        self.VA_sheet = xls.parse(0, header=None)
        self.data = None

    # read chunks of size 6 from .csv
    def flow_from_df(self, chunk_size: int = 6):
        for start_row in range(0, self.VA_sheet.shape[0], chunk_size):
            end_row = min(start_row + chunk_size, self.VA_sheet.shape[0])
            yield self.VA_sheet.iloc[start_row:end_row, :]

    # convert VA to decimal representation
    def convert_Snellen_to_decimal(self, visualAcuity):
        # if interval choose median value
        if '-' in visualAcuity:
            f1, f2 = visualAcuity.split('-')
            if f1 is not '' and f2 is not '':
                num1, denom1 = f1.split('/')
                num2, denom2 = f2.split('/')
                visualAcuity = (float(num1) / int(denom1) + float(num2) / int(denom2)) / 2
        else:
            if visualAcuity is not '' and visualAcuity != '0':
                num, denom = visualAcuity.split('/')
                visualAcuity = float(num) / int(denom)

        return visualAcuity

    def get_visual_acuity_data(self):
        get_chunk = self.flow_from_df()
        chunk = next(get_chunk)

        data = pd.DataFrame()
        index = 0

        while True:
            try:
                patientID = chunk.iloc[1][1]

                if patientID == '252/1911':
                    patientID = '252'

                # perform data cleaning and solve inconsistencies
                for i in range(2, 19):
                    date = chunk.iloc[3][i]
                    visualAcuityR = chunk.iloc[4][i]
                    visualAcuityL = chunk.iloc[5][i]

                    if pd.isna(date) or date is None or date is '':
                        continue

                    # remove letters from visual acuity fields and convert fractions to float
                    visualAcuityR = re.sub('[^0-9/-]', '', str(visualAcuityR))
                    visualAcuityL = re.sub('[^0-9/-]', '', str(visualAcuityL))

                    visualAcuityR = self.convert_Snellen_to_decimal(visualAcuityR)
                    visualAcuityL = self.convert_Snellen_to_decimal(visualAcuityL)

                    if visualAcuityL is not '':
                        newEntryL = pd.DataFrame({'ID': [str(patientID) + 'OS'],
                                                  'Date': [date],
                                                  'VA': [visualAcuityL]})
                        data = data.append(newEntryL)
                        index += 1

                    if visualAcuityR is not '':
                        newEntryR = pd.DataFrame({'ID': [str(patientID) + 'OD'],
                                                  'Date': [date],
                                                  'VA': [visualAcuityR]})
                        data = data.append(newEntryR)
                        index += 1
                #print(chunk)
                chunk = next(get_chunk)
            except StopIteration:
                break

        data = data.set_index('ID')
        data = data.sort_values(['ID', 'Date'], ascending=[True, True])
        self.data = data

    def format_timestamps(self):
        # the time intervals can be represented in number of days/weeks/months
        # best choice would be months according to ophthalmologists
        # denominator = 7 => nb of weeks, denominator = 1 => nb_of days, denominator = 30 => nb of months
        groups = self.data.groupby(['ID'])
        self.data['Timestamp'] = groups.transform('min')
        self.data['Timestamp'] = (self.data['Date'] - self.data['Timestamp']).dt.days/30
        self.data['Timestamp'] = self.data['Timestamp'].values.astype(np.float)
        self.data['VA'] = self.data['VA'].values.astype(np.float)

        self.data['ID'] = self.data.index
        self.data.index = [self.data['ID'], self.data['Date']]
        del self.data['Date']
        del self.data['ID']

    def resample_time_series(self):
        # mean_len = 0
        # nb_series = len(self.data.groupby('ID'))
        # for entrynb, entry in self.data.groupby('ID'):
        #     mean_len += entry.index.size
        # mean_len = round(float(mean_len)/nb_series)
        self.data = self.data.reset_index(level='ID')
        self.data = self.data.groupby('ID').resample('M').mean().interpolate()
        del self.data['Timestamp']


if __name__ == '__main__':
    data_builder = DatasetBuilder()
    data_builder.get_visual_acuity_data()
    data_builder.format_timestamps()
    #print(data_builder.data)
    #data_builder.data.plot.scatter(x='Timestamp', y='VA')
    #plt.show()

    data_builder.resample_time_series()
    print(data_builder.data)

    svr_regression(data_builder.data)
    #dtw_clustering(data_builder.data, 2)

