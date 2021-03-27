import re

import pandas as pd
import matplotlib.pyplot as plt

# reorganise and clean data
from experiments.image_analysis import ImageAnalysis


class VisualAcuityAnalysis:

    def __init__(self):
        self.img_analysis = ImageAnalysis()

    # read chunks of size 3 from .csv
    def flow_from_df(self, dataframe: pd.DataFrame, chunk_size: int = 3):
        for start_row in range(0, dataframe.shape[0], chunk_size):
            end_row = min(start_row + chunk_size, dataframe.shape[0])
            yield dataframe.iloc[start_row:end_row, :]

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
            if visualAcuity is not '' and visualAcuity is not '0':
                num, denom = visualAcuity.split('/')
                visualAcuity = float(num) / int(denom)

        return visualAcuity

    def get_visual_acuity_data(self):
        inputData = pd.read_csv("/data_layer/DMLVAVcuID.csv", nrows=282, header=None)

        get_chunk = self.flow_from_df(inputData)
        chunk = next(get_chunk)

        data = pd.DataFrame()
        index = 0

        while True:
            try:
                entryNumber = chunk[0].iloc[0]
                patientID = chunk[1].iloc[0]

                # perform data cleaning and solve inconsistencies

                if patientID == '8615':
                    patientID = '276'

                if patientID == '252/1911':
                    patientID = '252'

                for i in range(2, 19):
                    dateString = chunk[i].iloc[0]
                    visualAcuityR = chunk[i].iloc[1]
                    visualAcuityL = chunk[i].iloc[2]

                    if pd.isna(dateString) or pd.to_datetime(dateString, format='%d.%m.%Y', errors='coerce') is pd.NaT:
                        break

                    # remove letters from visual acuity fields and convert fractions to float
                    visualAcuityR = re.sub('[^0-9/-]', '', str(visualAcuityR))
                    visualAcuityL = re.sub('[^0-9/-]', '', str(visualAcuityL))

                    visualAcuityR = self.convert_Snellen_to_decimal(visualAcuityR)
                    visualAcuityL = self.convert_Snellen_to_decimal(visualAcuityL)

                    newEntry = pd.DataFrame({'Nr': [entryNumber],
                                             'ID': [patientID],
                                             'Date': [dateString],
                                             'Right eye': [visualAcuityR],
                                             'Left eye': [visualAcuityL]})
                    data = data.append(newEntry)
                    index += 1
                # print(chunk)
                chunk = next(get_chunk)
            except StopIteration:
                break

        data = data.set_index('ID')
        return data

    # split left and right eye data
    def split_eye_data(self, data):

        data['Date']= pd.to_datetime(data['Date'], format='%d.%m.%Y')
        data['Left eye']= pd.to_numeric(data['Left eye'])
        data['Right eye']= pd.to_numeric(data['Right eye'])
        data['Nr'] = pd.to_numeric(data['Nr'])

        leftEyeData = pd.DataFrame({
            "New ID": data['Nr'],
            "Date": data['Date'],
            'Eye': data['Left eye']
        })

        rightEyeData = pd.DataFrame({
            "New ID": data['Nr'],
            "Date": data['Date'],
            'Eye': data['Right eye']
        })

        return leftEyeData, rightEyeData

    def format_append_data(self, eyeDf, dataList):
        for entrynb, entry in eyeDf.groupby(['New ID']):
            if entry.index.size != 1:
                entry = entry.sort_values(by=['Date'])
                firstVisit = int(entry.iloc[0]['Date'].timestamp() // (60 * 60 * 24))
                # nb of days from the first visit?
                # denominator = 7 => nb of weeks, denominator = 1 => nb_of days, denominator = 30 => nb of months
                entry['New Date'] = (entry['Date'].astype('int64') // (60 * 60 * 24 * (10 ** 9)) - firstVisit)/7
                dataList.append(entry)

    # visualize time series with 2-class labels
    def plot_time_series(self, eyeData, labels=None):
        fig, ax = plt.subplots()
        fig.set_size_inches(16, 10.5)

        plt.style.use('fivethirtyeight')
        cmap = ['red', 'blue']

        for i in range(len(eyeData)):
            nb_visits = eyeData[i].shape[0]
            if nb_visits > 10:
                visits = []
                for visit in range(nb_visits):
                    visits.append(visit)
                id = eyeData[i]['New ID'].tolist()[0]
                if labels is not None:
                    label = labels[id]
                    color = cmap[label]
                else:
                    color = cmap[0]
                plt.plot(visits, eyeData[i]['Eye'], color=color, alpha=0.5)

        plt.show()

    # set entry ID based on patient ID and eye (left or right)
    def set_new_ID(self, eyeData, eyeString):
        index_list = eyeData.index.tolist()
        new_id_list = []
        for index in index_list:
            new_id = index + eyeString
            new_id_list.append(new_id)
        eyeData['New ID'] = new_id_list

    # compare actual VA labels with labels obtained from clustering
    def compare_labels(self, eyeData, labels):
        accuracy = 0
        nb_series = 0
        for df in eyeData:
            actual_label = 0
            if df['Eye'].iloc[0] >= df['Eye'].iloc[df.shape[0] - 1]:
                actual_label = 1
            id = df['New ID'].iloc[0]
            if id in labels:
                obtained_label = labels[id]
                if obtained_label == actual_label:
                    accuracy += 1
                nb_series += 1

        accuracy = float(accuracy)/nb_series
        print("clustering accuracy: " + str(accuracy))

    # visualize cluster distribution
    def clusters_distribution(self, obtainedLabels):
        good = 0
        for l in obtainedLabels.values():
            if l == 0:
                good += 1

        bad = len(obtainedLabels) - good
        good = float(good) / len(obtainedLabels)
        bad = float(bad) / len(obtainedLabels)
        print(str(good) + " " + str(bad))

        labels = ['Good evolution', 'Bad evolution']
        percentages = []
        percentages.append(good)
        percentages.append(bad)
        plt.bar(labels, percentages)

        plt.savefig("plots/evolution_distribution")

    # get the final dataframe
    def get_va_df(self):
        data = self.get_visual_acuity_data()

        leftEyeData, rightEyeData = self.split_eye_data(data)

        self.set_new_ID(leftEyeData, "OS")
        self.set_new_ID(rightEyeData, "OD")

        eyeData = []
        self.format_append_data(leftEyeData, eyeData)
        self.format_append_data(rightEyeData, eyeData)

        return eyeData

    # get the list with all visual acuity sequences
    def get_va_feature_list(self, eyeData):
        sequences = []
        for df in eyeData:
            sequence = []
            for i in range(len(df.index)):
                sequence.append(df['Eye'].iloc[i])
            sequences.append(sequence)

        return sequences


if __name__ == '__main__':
    va_analysis = VisualAcuityAnalysis()

    eyeData = va_analysis.get_va_df()

    print(eyeData)

    #obtainedLabels = va_analysis.img_analysis.get_labels_dictionary()

    #va_analysis.compare_labels(eyeData, obtainedLabels)
    #va_analysis.clusters_distribution(obtainedLabels)

    #va_analysis.plot_time_series(eyeData)
