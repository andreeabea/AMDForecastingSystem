import shap
import tensorflow as tf
import numpy as np

from data_handling.db_handler import DbHandler
from data_handling.timeseries_augmentation import TimeSeriesGenerator
from neural_networks.rnn import Rnn
from regression import TimeSeriesRegressor


class Explainer:

    def __init__(self, data, include_timestamp=False, previous_visits=3, features='exclude VA'):
        self.data = data
        self.include_timestamp = include_timestamp
        gen = TimeSeriesGenerator(data)

        tf.compat.v1.disable_v2_behavior()
        dependencies = {
            'root_mean_squared_error': Rnn.root_mean_squared_error
        }
        self.model = tf.keras.models.load_model("./best_models/gru-num-notres-0.80.h5", custom_objects=dependencies)
        X, Y = gen.generate_timeseries(include_timestamp, previous_visits, features)
        self.X = X
        trainX, trainY, validX, validY, testX, testY = TimeSeriesRegressor.train_test_val_split(X, Y)
        self.rnn = Rnn(trainX, trainY, validX, validY, testX, testY, model=self.model)
        #self.rnn.train()
        #self.rnn.evaluate_model()

    def explain_rnn_prediction(self):
        explainer = shap.DeepExplainer(self.rnn.model, self.X)
        shap_values = explainer.shap_values(self.X)

        shap.initjs()
        #shap.plots.beeswarm(shap_values)
        features1 = self.data.columns.values
        features1 = features1[1:]
        lambda_expr = lambda x: x + ' visit 2'
        features2 = lambda_expr(features1)

        if include_timestamps:
            features2 = np.append(features2, "Prediction timestep")

        features = np.concatenate((features1, features2))

        # plot the explanation of the first prediction
        # the model is "multi-output" because it is rank-2 but only has one column
        shap.save_html("shap.html", shap.force_plot(explainer.expected_value[0], shap_values[0][0],
                                                    features=features,
                                                    link='logit'))


if __name__ == '__main__':
    datatype = 'all'
    include_timestamps = False

    db_handler = DbHandler(datatype, include_timestamps)
    data = db_handler.get_data_from_csv()

    explainer = Explainer(data, include_timestamps)
    explainer.explain_rnn_prediction()
