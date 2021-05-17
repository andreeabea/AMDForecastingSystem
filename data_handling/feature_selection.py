import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold, cross_val_score


class FeatureSelector:

    def __init__(self, data, gen):
        self.data = data
        self.gen = gen

    def rfe(self, datatype='all', include_timestamp=False, previous_visits=1, features='exclude VA'):
        X, Y = self.gen.generate_timeseries(include_timestamp, previous_visits, features)
        print(mutual_info_regression(X, Y))
        model = GradientBoostingRegressor()
        # for numerical and image data 28 - 53.21
        # gru - 53.13
        if datatype == 'numerical':
            # 11 best for numerical data
            n_features_to_select = 11
        else:
            n_features_to_select = 28
        model = RFE(model, n_features_to_select=n_features_to_select)
        cv = KFold(n_splits=10)
        n_scores = cross_val_score(model, X, Y, cv=cv, n_jobs=-1)
        print('R^2: ' + str(np.mean(n_scores)))
        fit = model.fit(X, Y)
        # plot_feature_importances(fit, X.shape[1])
        print("Num Features: %d" % fit.n_features_)
        print("Selected Features: %s" % fit.support_)
        print("Feature Ranking: %s" % fit.ranking_)

        feature_vector = []
        for i in range(len(fit.support_)):
            if fit.support_[i]:
                feature_vector.append(i + 1)
        print(feature_vector)

        return feature_vector

    def lasso_feature_selector(self, include_timestamp=False, features='exclude VA'):
        X, Y = self.gen.generate_timeseries(self.data, include_timestamp, 1, features)

        lasso = LassoCV(normalize=True)
        lasso.fit(X, Y)
        print("Best alpha using built-in LassoCV: %f" % lasso.alpha_)
        print("Best score using built-in LassoCV: %f" % lasso.score(X, Y))
        importance_vector = lasso.coef_
        print(importance_vector)

        feature_vector = []
        for i in range(len(importance_vector)):
            if importance_vector[i] != 0:
                feature_vector.append(i + 1)
        print(feature_vector)

        return feature_vector
