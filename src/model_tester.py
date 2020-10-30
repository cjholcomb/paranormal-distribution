from columns_wrangling import *
import re

import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def train_split(df, target, feature_list = model_features_basic):
    '''
    Imports and splits the dataset according to designated target.

    References in test_regressors and test_classifiers

    params -- target, str, one of ['retweets', 'favorites']
    returns -- four dataframes, X_train, X_test, y_train, y_test'''

    X = df.loc[:,feature_list].values
    y_target = df.loc[:,[target]].values * 1
    y_target = y_target.reshape(len(y_target),)
    X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def corr_matrix(df, feature_list = model_features_basic, cmap = 'seismic_r', center = 0):
    ''' produces a correlation matrix heatmap of features to be included in the model.

    params:  

    df           -- pandas dataframe
    feature_list -- list of featues to be included in the model. default model_features_basic
    cmap         -- matplotlib colormap, default seismic(reversed)
    center       -- float, between [-1, 1]. center point of colormap
    
    returns:
    axes object'''

    sns.set_style('darkgrid')
    corr = df[feature_list].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots()
    ax = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, 
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, robust = True)
    plt.show();
    return ax

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
    '''creates a list of models with optimized hyperparamters.
    parameters tuned elsewhere- this function is for evaluating tuned models.
    hyperparameters must be added within function code

    Arguments:
    none

    Returns:
    list of models'''
    
    models = []
    models.append(XGB.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, eta=0.1, gamma=0,
              importance_type='gain', learning_rate=0.1,
              max_delta_step=0, max_depth=10, min_child_weight=5, missing=None,
              n_estimators=100, n_jobs=-1, nthread=None, objective='reg:squarederror',
              random_state=7, reg_alpha=0, reg_lambda=0.3,
              sampling_method='uniform', scale_pos_weight=1, seed=None,
              silent=None, subsample=0.7, tree_method='hist', verbosity=1))
    return models


def fit_models(models, target, X_train, X_test, y_train, y_test):
    ''' fits all models in list to train data

    Arguments:

    models -- list, must be instantiated models
    target -- str, must be 'retweets' or 'favorites'
    X_train, X_test, y_train, y_test -- numpy arrays, derived from train_test_split
    
    Returns:

    list of fit models '''
    for model in list(models):
        model.fit(X_train, y_train)
    return models

def test_regressors(models, target, metric, X_train, X_test, y_train, y_test):
    ''' evaluates all models on all sklearn scoring metrics.

    Arguments:

    models -- list, must be fitted models
    target -- str, must be 'retweets' or 'favorites'
    metric -- evaluation metric, must be imported from sklearn.metrics

    Returns:

    dictionary, model is key, scoring metric is value'''
    
    dct = {}  
    for model in models:
        dct[str(metric)] = metric(y_test, model.predict(X_test))
    return dct

def model_comparison(results, title = ''):
    ''' produces barchart of model evaluation metrics

    Arguments:

    results -- dictionary. Keys must be models, values must be metrics
    title   -- str, title of chart. default empty

    Returns:

    Barchart'''

    sns.set_style('darkgrid')
    fig, ax = plt.subplots()
    ax = sns.barplot(x= list(results.keys()), y =list(results.values()))
    ax.set_xticklabels(labels = results.keys(), rotation = 90)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title(title)
    plt.show();

def feature_importance(model, feature_list = model_features_basic, title = 'Feature Importance'):
    ''' produces barchart of model evaluation metrics

    Arguments:

    model -- fitted model
    feature_list -- list of featues to be included in the model. default model_features_basic
    title   -- str, title of chart. 'Feature Importance'


    Returns:
    Barchart'''
    sns.set_style('darkgrid')
    fig, ax = plt.subplots()
    ax = sns.barplot(feature_list, model.feature_importances_)
    ax.set_xticklabels(labels = model_features_basic, rotation = 90)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title(title)
    plt.show();

if __name__ == '__main__':
    pass


