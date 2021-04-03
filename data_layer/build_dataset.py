import re

import pandas as pd
import numpy as np
import xlrd
from xlutils.copy import copy
from xlwt import Style
import matplotlib.pyplot as plt

from oct_feature_extraction import OCTFeatureExtractor


class DatasetBuilder:

    def __init__(self, path):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.options.display.float_format = '{:,.2f}'.format

        # DatasetBuilder.handle_bold_cells(path)
        xls = pd.ExcelFile(path)
        self.VA_sheet = xls.parse(0, header=None)
        self.data = pd.DataFrame()
        self.retina_features = []

    @staticmethod
    def handle_bold_cells(path):
        workbook = xlrd.open_workbook(path, formatting_info=True)
        sheet = workbook.sheet_by_index(0)
        new_wb = copy(workbook)
        for row in range(0, sheet.nrows):
            for col in range(0, sheet.ncols):
                cell = sheet.cell(row, col)
                format = workbook.xf_list[cell.xf_index]
                if workbook.font_list[format.font_index].weight == 700:
                    print(str(cell.value) + " t")
                    new_wb.get_sheet(0).write(row, col, str(cell.value) + " t", Style.easyxf("font: bold on;"))

        new_wb.save(path)

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
            if visualAcuity is not '' and visualAcuity != '0' and visualAcuity != 't':
                num, denom = visualAcuity.split('/')
                visualAcuity = float(num) / int(denom)

        return visualAcuity

    def check_treatment(self, visualAcuity):
        try:
            if pd.notna(visualAcuity) and 't' in visualAcuity:
                treatment = 1
            else:
                treatment = 0
        except TypeError:
            treatment = 0
        return treatment

    def build_visual_acuity_df(self):
        get_chunk = self.flow_from_df()
        chunk = next(get_chunk)

        data = pd.DataFrame()
        index = 0

        while True:
            try:
                patientID = chunk.iloc[1][1]

                if patientID == 8615:
                    patientID = '276'

                if patientID == '252/1911':
                    patientID = '252'

                # perform data cleaning and solve inconsistencies
                for i in range(2, 19):
                    date = chunk.iloc[3][i]
                    visualAcuityR = chunk.iloc[4][i]
                    visualAcuityL = chunk.iloc[5][i]

                    if pd.isna(date) or date is None or date is '':
                        continue

                    treatmentR = self.check_treatment(visualAcuityR)
                    treatmentL = self.check_treatment(visualAcuityL)

                    # remove letters from visual acuity fields and convert fractions to float
                    visualAcuityR = re.sub('[^0-9/-]', '', str(visualAcuityR))
                    visualAcuityL = re.sub('[^0-9/-]', '', str(visualAcuityL))

                    visualAcuityR = self.convert_Snellen_to_decimal(visualAcuityR)
                    visualAcuityL = self.convert_Snellen_to_decimal(visualAcuityL)

                    if visualAcuityL is not '':
                        newEntryL = pd.DataFrame({'ID': [str(patientID) + 'OS'],
                                                  'Date': [date],
                                                  'VA': [visualAcuityL],
                                                  'Treatment': [treatmentL]})
                        data = data.append(newEntryL)
                        index += 1

                    if visualAcuityR is not '':
                        newEntryR = pd.DataFrame({'ID': [str(patientID) + 'OD'],
                                                  'Date': [date],
                                                  'VA': [visualAcuityR],
                                                  'Treatment': [treatmentR]})
                        data = data.append(newEntryR)
                        index += 1
                # print(chunk)
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
        self.data['Timestamp'] = groups.transform('min').loc[:, 'Date':'Date']
        self.data['Timestamp'] = (self.data['Date'] - self.data['Timestamp']).dt.days / 30
        self.data['Timestamp'] = self.data['Timestamp'].values.astype(np.float)
        self.data['VA'] = self.data['VA'].values.astype(np.float)

        if len(self.data.columns) == 4:
            self.data['ID'] = self.data.index
            self.data.index = [self.data['ID'], self.data['Date']]
            del self.data['ID']
        del self.data['Date']

    def resample_time_series(self):
        self.data = self.data.reset_index(level='ID')
        self.data = self.data.groupby('ID').resample('M').mean()
        self.data['Treatment'] = self.data['Treatment'].fillna(0)
        self.data = self.data.interpolate()
        del self.data['Timestamp']

    def find_mean_sequence_len(self):
        mean_len = 0
        nb_series = len(self.data.groupby('ID'))
        for entrynb, entry in self.data.groupby('ID'):
            mean_len += entry.index.size
        mean_len = round(float(mean_len) / nb_series)
        return mean_len

    def describe_dataset(self):
        summary = self.data.describe()

        fig, ax = plt.subplots()
        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')

        # Specify values of cells in the table
        ax.table(cellText=summary.values,
                  # Specify width of the table
                  colWidths=[0.3] * len(self.data.columns),
                  # Specify row labels
                  rowLabels=summary.index,
                  # Specify column labels
                  colLabels=summary.columns)

        fig.tight_layout()
        plt.show()

    def join_OCT_features(self, df):
        self.data = self.data.merge(df, how='left', on=['ID', 'Date'])
        self.data = self.data.sort_values(['ID', 'Date'], ascending=[True, True])
        self.data.index = [self.data['ID'], self.data['Date']]
        #del self.data['Date']
        del self.data['ID']

    def interpolate_OCT_features(self):
        for feature_name in self.retina_features:
            def function(group):
                first = group[feature_name].first_valid_index()
                last = group[feature_name].last_valid_index()
                sliced = group.loc[first:last, :]
                initial_index = sliced.index
                sliced.index = sliced.index.get_level_values(1)
                sliced = sliced.interpolate(method='time')
                sliced.index = initial_index
                return sliced

            self.data = self.data.groupby('ID').apply(function).reset_index(level=0, drop=True)
            # remove groups with 1 visit
            self.data = self.data.groupby('ID').filter(lambda x: 1 < len(x) != x[feature_name].isnull().sum())

    def add_retina_features(self):
        oct_feature_extractor = OCTFeatureExtractor()
        df = oct_feature_extractor.get_feature_sequences(OCTFeatureExtractor.VOLUME_FEATURE, 0)
        feature_name = OCTFeatureExtractor.VOLUME_FEATURE + str(0)
        self.retina_features.append(feature_name)
        self.join_OCT_features(df)
        del self.data['Date']

        df = oct_feature_extractor.get_feature_sequences(OCTFeatureExtractor.VOLUME_FEATURE, 1)
        feature_name = OCTFeatureExtractor.VOLUME_FEATURE + str(1)
        self.retina_features.append(feature_name)
        self.join_OCT_features(df)
