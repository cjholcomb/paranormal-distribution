from src.columns_wrangling import *

import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


import xgboost as XGB

from sklearn.linear_model import LinearRegression as LRR
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.svm import SVR

from sklearn.metrics import r2_score, explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, mean_poisson_deviance, mean_gamma_deviance

def train_split(df, target):
    '''
    Imports and splits the dataset according to designated target.

    References in test_regressors and test_classifiers

    params:target, str, one of ['recovery', 'delta', 'decline']
    returns: four dataframes, X_train, X_test, y_train, y_test'''

    X = df.loc[:,features].values
    y_target = df.loc[:,[target]].values * 1
    y_target = y_target.reshape(len(y_target),)
    X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def instan_baseline_regressors():
    ''' creates seven types of regressor models with default parameters. Used for comparison to tuned models.
    Intended for delta or decline as target.

    params = None
    returns = list of instanstiated classifier models.'''
    models = []
    models.append(LRR())
    models.append(DTR())
    models.append(KNR())
    models.append(RFR())
    models.append(GBR())
    models.append(SVR())
    models.append(XGB.XGBRegressor())
    return models


def instan_tuned_regressors():
    ''' Placeholder for tuned regressor models'''
    pass


def fit_models(models, target, X_train, X_test, y_train, y_test):
    ''' fits all models in list to train data

    Arguments:

    models -- list, must be instantiated models
    target -- str, must be 'retweets' or 'favorites'
    X_train, X_test, y_train, y_test -- numpy arrays, derived from train_test_split
    
    Returns:

    list of fit models '''
    for model in models:
        model.fit(X_train, y_train)
    return models

def test_regressors(models, target, X_train, X_test, y_train, y_test):
    ''' evaluates all models on all sklearn scoring metrics.

    Arguments:

    models -- list, must be fitted models
    target -- str, must be 'retweets' or 'favorites'

    Returns:

    Nested dictionary. Scoring metrics dictionaries nested in model dictionary.'''

    dct = {}  
    y_predict = model.predict(X_test)
    for model in models:
        dct[model] = {}
        dct[model]['Exp Var'] = explained_variance_score(y_test, model.predict(X_test))
        dct[model]['Max Er'] = max_error(y_test, model.predict(X_test))
        dct[model]['Mean Abs Er'] = mean_absolute_error(y_test, model.predict(X_test))
        dct[model]['Mean Sq Er'] = mean_squared_error(y_test, model.predict(X_test))
        dct[model]['Mean Sq Log Er'] = mean_squared_log_error(y_test, model.predict(X_test))
        dct[model]['Med Abs Er'] = median_absolute_error(y_test, model.predict(X_test))
        dct[model]['r2'] = r2_score(y_test, model.predict(X_test))
        dct[model]['Mean Pois Dev'] = mean_poisson_deviance(y_test, model.predict(X_test))
        dct[model]['Mean Gamma Dev'] = mean_gamma_deviance(y_test, model.predict(X_test))
    return dct

# if __name__ == '__main__':
#     X_train, X_test, y_train, y_test = train_split('recovery')


