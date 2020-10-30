import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import plot_roc_curve
from scipy.stats import (mannwhitneyu, chisquare, fisher_exact)

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import NMF, PCA

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (confusion_matrix, precision_score, 
                                    recall_score, accuracy_score, r2_score, 
                                    plot_confusion_matrix)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, plot_confusion_matrix
from sklearn.cluster import DBSCAN

# data preparation as feature engineering for wine dataset
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from sklearn import (
    cluster, decomposition, ensemble, manifold, 
    random_projection, preprocessing)
from sklearn.utils.class_weight import compute_class_weight

from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

# from sklearn.neighbors import KNeighborsRegressor

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import random

import json


def find_top_n(n, df , column = 'tweet_text', feature = '#'):
    ''' 
    Parameters
    ----------
    n: number of top features to view
    df: dataframe
    column: column of interest to loop through
    feature: what to look for in column like, # or @

    Returns
    -------
    top_tags: sorted words that are most common
    top_counts: sorted total counts in order of top tags
    '''    
    d = {}
    for text in df[column]:
        #avoid the tweets that are already retweets
        if text[:3] == 'RT ':
            pass
        elif feature in text:
            lst = text.split()
            ats = [at for at in lst if at[0] == feature]
            for tags in ats:
                tags = tags.replace(':', '')
                if tags not in d:
                    d[tags] = 1
                else:
                    d[tags] += 1
        else:
            pass

    tags = np.array(list(d.keys()))
    counts = np.array(list(d.values()))
    top_tags = tags[counts.argsort()[::-1]][:n]
    top_counts = counts[counts.argsort()[::-1]][:n]
    return top_tags, top_counts


def one_hot_encode_term(top_tags, df, n = 10, column = 'tweet_text'):
    '''
    Parameters
    ----------
    top_tags: tags to one hot encode in dataframe
    n: how many you want to one hot encode
    df: dataframe
    column: column in df to loop through, checking if top tag is in the text

    Returns
    -------
    new dataframe with onehot encoded columns
    '''

    mentions = []
    for i in top_tags:
        for text in df[column]:
            mentions.append((i in text)*1)
            
    slices = np.linspace(len(df[column]), len(mentions), n)

    for i in range(len(top_tags)):
        if i == 0:
            df[f'mentions_{top_tags[i]}'] = mentions[:int(slices[i])]
        else:
            df[f'mentions_{top_tags[i]}'] = mentions[int(slices[i-1]):int(slices[i])]
    return df


def barplot_of_top_n(top_tags, top_counts, save = False, png_name = '../images/top_ten_hashtags.png', title = 'Top 25 Hashtags'):
    fig, ax = plt.subplots(figsize = (20,10))
    sns.barplot(top_tags, top_counts, palette='coolwarm')
    plt.xticks(rotation=70)
    plt.title(title)
    if save:
        plt.savefig(png_name)
    plt.show();


if __name__ == '__main__':
    df = pd.read_json('data/basic_dataset.json')

    top_hashtags, top_h_counts = find_top_n(n=10, df= df , column = 'tweet_text', feature = '#')
    top_mentions, top_m_counts = find_top_n(n=10, df= df , column = 'tweet_text', feature = '@')
    # barplot_of_top_n(top_hashtags, top_h_counts)
    # barplot_of_top_n(top_mentions, top_m_counts)
    #only view non retweeted info

    non_rt_df = df[df['is_retweet'] == 0]
    new_df = one_hot_encode_term(top_hashtags, n = 10, df = df, column = 'tweet_text')
    new_df = one_hot_encode_term(top_mentions, n = 10, df = df, column = 'tweet_text')