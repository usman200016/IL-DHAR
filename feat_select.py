import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import RandomForestClassifier

def recursive_feature_elimination(X, y, n_features_to_select):
    estimator = RandomForestClassifier()

    rfe_selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
    rfe_selector = rfe_selector.fit(X, y)

    selected_features_mask = rfe_selector.support_
    selected_features = X[:, selected_features_mask]

    return selected_features