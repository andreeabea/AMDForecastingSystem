import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from sklearn import linear_model, metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor, ExtraTreesRegressor, \
    RandomForestRegressor, BaggingRegressor, StackingRegressor
from sklearn.feature_selection import RFE, mutual_info_regression, RFECV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from tslearn.svm import TimeSeriesSVR
from tslearn.utils import to_time_series_dataset

import config
from data_layer.build_dataset import DatasetBuilder
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt

from neural_networks.lstm import Lstm

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:,.2f}'.format


def split_data(data):
    split_index = len(data.groupby('ID')) * 0.8
    i = 0
    trainX = []
    trainY = []
    testX = []
    testY = []
    for nb, group in data.groupby('ID'):
        j = group.index.size - 1
        to_predict = group.iloc[j]['VA']
        if i < split_index:
            trainX.append(group.head(j))
            trainY.append(to_predict)
        else:
            testX.append(group.head(j))
            testY.append(to_predict)
        i += 1

    trainX = pd.concat(trainX)
    testX = pd.concat(testX)

    trainX = trainX.groupby('ID').apply(pd.DataFrame.to_numpy).to_numpy().tolist()
    trainX = to_time_series_dataset(trainX)
    trainX = np.array(trainX)

    testX = testX.groupby('ID').apply(pd.DataFrame.to_numpy).to_numpy().tolist()
    testX = to_time_series_dataset(testX)

    trainY = np.array(trainY)
    testY = np.array(testY)

    return trainX, trainY, testX, testY


def svr_regression_tslearn(data):
    svr = TimeSeriesSVR(kernel="gak", gamma="auto")
    trainX, trainY, testX, testY = split_data(data)
    print(svr.fit(trainX, trainY).score(testX, testY))


def generate_timeseries(data, include_timestamp=False, size=1, features='all'):
    X = []
    Y = []

    sequences = data.groupby('ID').apply(pd.DataFrame.to_numpy).to_numpy().tolist()
    for i in range(len(sequences)):
        if len(sequences[i]) > size:
            generator = TimeseriesGenerator(sequences[i], sequences[i], length=size)
            x, y = generator[0]
            X.append(x)
            Y.append(y)

    X = np.vstack(X)
    Y = np.vstack(Y)

    # select most important features: 45% acc?
    feature_list = [22, 23, 17, 21, 11, 12, 7, 20, 16]

    if include_timestamp:
        # update the timestamps
        for i in range(len(X)):
            min_timestamp = X[i, 0, X.shape[2]-1]
            X[i, :, X.shape[2]-1] = np.subtract(X[i, :, X.shape[2]-1], min_timestamp)
            Y[i, Y.shape[1]-1] = Y[i, Y.shape[1]-1] - min_timestamp
        # split target visual acuity and move the future timestamp to X
        future_timestamp_array = Y[:, Y.shape[1] - 1:]
        # append the timestamp to the feature list
        feature_list.append(24)
    Y = Y[:, :1]
    if features == 'exclude VA':
        X = X[:, :, 1:]
    else:
        if features != 'all':
            if features == 'default':
                X = X[:, :, feature_list]
            else:
                if 25 in features:
                    X = X[:, :, features[:-1]]
                else:
                    X = X[:, :, features]
    X = X.reshape(-1, X.shape[1] * X.shape[2])
    if include_timestamp and future_timestamp_array.shape[1] != 0:
        X = np.hstack((X, future_timestamp_array))

    #pca = PCA(16)
    #X = pca.fit_transform(X)
    #tsne = TSNE(n_components=3, verbose=1, perplexity=25, n_iter=1000, learning_rate=0.1)
    #X = tsne.fit_transform(X)
    return X, Y.reshape(-1)


def train_test_val_split(dataX, dataY):
    # compute the training split
    i = int(len(dataX) * config.TRAIN_SPLIT)
    trainX = dataX[:i]
    trainY = dataY[:i]

    # obtain the validation and testing splits
    valid = int(len(dataX) * config.VAL_SPLIT)
    validX = dataX[i:i + valid]
    validY = dataY[i:i + valid]

    testX = dataX[i + valid:]
    testY = dataY[i + valid:]

    # return trainX.reshape(-1, trainX.shape[1]*trainX.shape[2]), trainY.reshape(-1), \
    #        testX.reshape(-1, testX.shape[1]*testX.shape[2]), testY.reshape(-1)
    return trainX, trainY, validX, validY, testX, testY


def svr_regression(data):
    X, Y = generate_timeseries(data, False)

    gbr = SVR()
    cv = KFold(n_splits=10)
    n_scores = cross_val_score(gbr, X, Y, cv=cv, n_jobs=-1)
    print('Accuracy: ' + str(np.mean(n_scores)))
    plot_feature_importances(gbr.fit(X, Y), X.shape[1])


def gradient_boosted_regression(data, include_timestamp=False, previous_visits=1, features='all'):
    X, Y = generate_timeseries(data, include_timestamp, previous_visits, features)
    gbr = GradientBoostingRegressor()

    cv = KFold(n_splits=10)
    n_scores = cross_val_score(gbr, X, Y, cv=cv, n_jobs=-1)
    print('Accuracy: ' + str(np.mean(n_scores)))
    fit = gbr.fit(X, Y)
    plot_feature_importances(fit, X.shape[1])


def feature_selector(data, include_timestamp=False, previous_visits=1, features='exclude VA'):
    X, Y = generate_timeseries(data, include_timestamp, previous_visits, features)
    print(mutual_info_regression(X, Y))
    gbr = GradientBoostingRegressor()
    # 11 best
    gbr = RFE(gbr, n_features_to_select=11)
    cv = KFold(n_splits=10)
    n_scores = cross_val_score(gbr, X, Y, cv=cv, n_jobs=-1)
    print('Accuracy: ' + str(np.mean(n_scores)))
    fit = gbr.fit(X, Y)
    # plot_feature_importances(fit, X.shape[1])
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)
    feature_vector = []
    for i in range(len(fit.support_)):
        if fit.support_[i]:
            feature_vector.append(i+1)
    print(feature_vector)
    return feature_vector


def extratrees_regression(data, include_timestamp=False):
    X, Y = generate_timeseries(data, include_timestamp)

    gbr = ExtraTreesRegressor()
    cv = KFold(n_splits=10)
    n_scores = cross_val_score(gbr, X, Y, cv=cv, n_jobs=-1)
    print('Accuracy: ' + str(np.mean(n_scores)))
    plot_feature_importances(gbr.fit(X, Y), X.shape[1])


def lasso_regression(data, include_timestamp=False, previous_visits=1, features='exclude VA'):
    X, Y = generate_timeseries(data, include_timestamp, previous_visits, features)
    clf = linear_model.Lasso(alpha=0.1)
    cv = KFold(n_splits=10)
    n_scores = cross_val_score(clf, X, Y, cv=cv, n_jobs=-1)
    print('Accuracy: ' + str(np.mean(n_scores)))


def voting_regression(data, include_timestamp=False, previous_visits=1, features='exclude VA'):
    gbr = GradientBoostingRegressor()

    X, Y = generate_timeseries(data, include_timestamp, previous_visits, features)
    X = X.reshape(-1, previous_visits, X.shape[1])

    def build_lstm():
        lstm = Lstm(None, None, None, None, None, None, timesteps=previous_visits, nb_features=X.shape[2])
        lstm.model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        return lstm.model

    keras_lstm = tf.keras.wrappers.scikit_learn.KerasRegressor(build_lstm, epochs=50, verbose=False)
    keras_lstm._estimator_type = 'regressor'

    vr = VotingRegressor([('gb', gbr), ('lstm', keras_lstm)])
    cv = KFold(n_splits=10)
    n_scores = cross_val_score(vr, X, Y, cv=cv, n_jobs=-1)
    print('Accuracy: ' + str(np.mean(n_scores)))


def plot_feature_importances(regressor, nb_features):
    feature_names = list(range(0, nb_features))
    sorted_ids = np.argsort(regressor.feature_importances_)
    pos = np.arange(sorted_ids.shape[0]) + .5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, regressor.feature_importances_[sorted_ids], align='center')
    plt.yticks(pos, np.array(feature_names)[sorted_ids])
    plt.title('Feature Importance')
    plt.show()


def lstm_regression(data, include_timestamp=False, previous_visits=1, features='exclude VA'):
    X, Y = generate_timeseries(data, include_timestamp, previous_visits, features)
    trainX, trainY, validX, validY, testX, testY = train_test_val_split(X, Y)

    lstm = Lstm(trainX, trainY, validX, validY, testX, testY)
    lstm.train()

    lstm.evaluate_model()

    # explainer = shap.DeepExplainer(lstm.model, trainX)
    # shap_values = explainer.shap_values(testX)
    #
    # shap.initjs()
    #
    # # plot the explanation of the first prediction
    # # Note the model is "multi-output" because it is rank-2 but only has one column
    # shap.force_plot(explainer.expected_value[0], shap_values[0][0])


def lstm_ensemble_regression(data, include_timestamp=False, previous_visits=1, features='exclude VA'):
    X, Y = generate_timeseries(data, include_timestamp, previous_visits, features)
    X = X.reshape(-1, previous_visits, X.shape[1])

    def build_lstm():
        lstm = Lstm(None, None, None, None, None, None, timesteps=previous_visits, nb_features=X.shape[2])
        lstm.model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        return lstm.model

    keras_lstm = tf.keras.wrappers.scikit_learn.KerasRegressor(build_lstm, epochs=50, verbose=False)
    keras_lstm._estimator_type = 'regressor'

    lstm_ensemble = AdaBoostRegressor(base_estimator=keras_lstm)
    cv = KFold(n_splits=10)
    n_scores = cross_val_score(lstm_ensemble, X, Y, cv=cv, n_jobs=-1)
    print('Accuracy: ' + str(np.mean(n_scores)))
    #plot_feature_importances(lstm_ensemble.fit(X, Y), X.shape[1])


def lasso_feature_selector(data, include_timestamp=False, features='exclude VA'):
    X, Y = generate_timeseries(data, include_timestamp, 1, features)
    #lasso_model = Lasso()
    #search = GridSearchCV(lasso_model,
    #                      {'alpha': np.arange(0.001, 0.1, 0.001)},
    #                      cv=5, scoring="neg_mean_squared_error", verbose=3
    #                      )
    # search.fit(X, Y)
    # print(search.best_params_)
    # coefficients = search.best_estimator_.coef_
    # importance = np.abs(coefficients)
    # print(importance)

    reg = LassoCV()
    reg.fit(X, Y)
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" % reg.score(X, Y))
    importance = reg.coef_


    feature_vector = []
    for i in range(len(importance)):
        if importance[i] != 0:
            feature_vector.append(i+1)
    print(feature_vector)
    return feature_vector


if __name__ == '__main__':
    #DatasetBuilder.write_all_data_to_csv("image_data.csv", datatype='images', include_timestamps=True)
    include_timestamps = True
    datatype = 'all'
    if include_timestamps:
        last_column = 'Timestamp'
        if datatype == 'numerical':
            data = pd.read_csv("all_data.csv", index_col=['ID', 'Date'], parse_dates=True)
        else:
            if datatype == 'images':
                data = pd.read_csv("image_data.csv", index_col=['ID', 'Date'], parse_dates=True)
            else:
                num_data = pd.read_csv("all_data.csv", index_col=['ID', 'Date'], parse_dates=True)
                img_data = pd.read_csv("image_data.csv", index_col=['ID', 'Date'], parse_dates=True)
                del num_data['Timestamp']
                del img_data['VA']
                del img_data['Treatment']
                data = num_data.merge(img_data, how='left', on=['ID', 'Date'])
    else:
        last_column = 'TotalVolume0'
        data = pd.read_csv("all_data_resampled.csv", index_col=['ID', 'Date'], parse_dates=True)


    data[data.columns] = MinMaxScaler().fit_transform(data)

    #voting_regression(data, include_timestamps, 1)
    #feature_vector = feature_selector(data, include_timestamps, 'exclude VA')
    gradient_boosted_regression(data, include_timestamps, 1, 'all')
