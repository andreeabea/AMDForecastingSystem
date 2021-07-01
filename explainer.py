import shap
import tensorflow as tf
import numpy as np

from data_handling.db_handler import DbHandler
from data_handling.timeseries_augmentation import TimeSeriesGenerator
from neural_networks.rnn import Rnn
from regression import TimeSeriesRegressor


class Explainer:

    def __init__(self, data, include_timestamp=False, previous_visits=2, features='exclude VA'):
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

    def explain_rnn_prediction(self):
        explainer = shap.DeepExplainer(self.rnn.model, self.rnn.trainX)
        shap_values = explainer.shap_values(self.rnn.testX)

        shap.initjs()
        features1 = self.data.columns.values
        features1 = features1[1:]
        lambda_expr = lambda x: x + ' visit 2'
        features2 = lambda_expr(features1)

        if include_timestamps:
            features2 = np.append(features2, "Prediction timestep")

        features = np.concatenate((features1, features2))

        # plot the explanation of the first prediction
        # the model is "multi-output" because it is a 2d array, but only has one column
        shap.save_html("shap2.html", shap.force_plot(explainer.expected_value[0], shap_values[0][0],
                                                    features=features,
                                                    link='logit'))


if __name__ == '__main__':
    datatype = 'numerical'
    include_timestamps = False

    db_handler = DbHandler(datatype, include_timestamps)
    data = db_handler.get_data_from_csv()

    explainer = Explainer(data, include_timestamps)
    explainer.explain_rnn_prediction()