import shap
import tensorflow as tf
import numpy as np

from data_processing.db_handler import DbHandler
from data_processing.timeseries_augmentation import TimeSeriesGenerator
from neural_networks.rnn import Rnn
from regression import TimeSeriesRegressor
from itertools import islice, chain


class Explainer:

    def __init__(self, data, include_timestamp=False, previous_visits=3, features='exclude VA'):
        self.data = data
        self.include_timestamp = include_timestamp
        gen = TimeSeriesGenerator(data)

        tf.compat.v1.disable_v2_behavior()
        tf.compat.v1.disable_eager_execution()

        #dependencies = {
        #    'root_mean_squared_error': Rnn.root_mean_squared_error
        #}
        #self.model = tf.keras.models.load_model("./best_models/lstm-num-notres-0.83.h5", custom_objects=dependencies)
        X, Y = gen.generate_timeseries(include_timestamp, previous_visits, features)
        trainX, trainY, validX, validY, testX, testY = TimeSeriesRegressor.train_test_val_split(X, Y)
        self.rnn = Rnn(trainX, trainY, validX, validY, testX, testY, nn_type='lstm')
        self.rnn.train()
        #self.rnn.evaluate_model()

    @staticmethod
    def slice_features(iterable, *selectors):
        return chain(*(islice(iterable, *s) for s in selectors))

    # TODO: select numerical features if all data is used
    def explain_rnn_prediction(self, previous_visits, features_str):
        explainer = shap.DeepExplainer(self.rnn.model, self.rnn.trainX)
        shap_values = explainer.shap_values(self.rnn.testX)

        shap.initjs()
        features1 = self.data.columns.values
        if features_str == 'exclude VA':
            features1 = features1[1:]

        for i in range(0, 23):
            if 'MinCentralThickness' in features1[i] or 'MaxCentralThickness' in features1[i] \
                    or 'CentralThickness' in features1[i] or 'TotalVolume' in features1[i]:
                features1[i] = features1[i][:len(features1[i]) - 1]
            else:
                if '0' in features1[i]:
                    features1[i] = features1[i][:len(features1[i]) - 1] + 'C0'
                elif '1' in features1[i]:
                    features1[i] = features1[i][:len(features1[i]) - 1] + 'N1'
                elif '2' in features1[i]:
                    features1[i] = features1[i][:len(features1[i]) - 1] + 'N2'
                elif '3' in features1[i]:
                    features1[i] = features1[i][:len(features1[i]) - 1] + 'S1'
                elif '4' in features1[i]:
                    features1[i] = features1[i][:len(features1[i]) - 1] + 'S2'
                elif '5' in features1[i]:
                    features1[i] = features1[i][:len(features1[i]) - 1] + 'T1'
                elif '6' in features1[i]:
                    features1[i] = features1[i][:len(features1[i]) - 1] + 'T2'
                elif '7' in features1[i]:
                    features1[i] = features1[i][:len(features1[i]) - 1] + 'I1'
                elif '8' in features1[i]:
                    features1[i] = features1[i][:len(features1[i]) - 1] + 'I2'

        #print(features1)
        features = []
        if previous_visits >= 2:
            lambda_expr = lambda x: x + ' visit 2'
            features2 = lambda_expr(features1)

            if include_timestamps and previous_visits == 2:
                features2 = np.append(features2, "Prediction timestep")

            features = np.concatenate((features1, features2))

        if previous_visits == 3:
            lambda_expr = lambda x: x + ' visit 3'
            features3 = lambda_expr(features1)

            if include_timestamps:
                features3 = np.append(features3, "Prediction timestep")

            features = np.concatenate((features, features3))

        # plot the explanation of the first prediction
        # the model is "multi-output" because it is a 2d array, but only has one column
        shap.save_html("shap_explanations/shap5.html", shap.force_plot(explainer.expected_value[0], shap_values[0][1],
                                                                       features=features,
                                                                       link='logit'))


if __name__ == '__main__':
    datatype = 'numerical'
    include_timestamps = False
    features = 'exclude VA'
    previous_visits = 3

    db_handler = DbHandler(datatype, include_timestamps)
    data = db_handler.get_data_from_csv()

    explainer = Explainer(data, include_timestamps, previous_visits, features)
    explainer.explain_rnn_prediction(previous_visits, features)
