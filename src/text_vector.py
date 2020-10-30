import pandas as pd 
import numpy as np

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import STOPWORDS

from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 

from sklearn.cluster import KMeans

def lemmatizer(string):
    ''' Lemmatize a string and return it in its original format 
    input: text string
    output: lemmatized text'''
    #iterate through string and pop all stop words 
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w)
                    for w in w_tokenizer.tokenize(string)
                    if "http" not in w])

def orig_vect(X_train, custom_stopwords):
    ''' pass in x training data to fit and transform
    input: text to be vectorized, customer list of stopwords
    output: vectorizer, x_train vectorized
    '''
    vectorizer = TfidfVectorizer(stop_words=custom_stopwords, max_features = 1000, analyzer='word', ngram_range=(1,2))
    x_train_vect = vectorizer.fit_transform(X_train)
    return vectorizer, x_train_vect

def lower_lemma(df, col):
    ''' lower and lemmatize the tweet 
    input: df = dataframe with tweet, col = column with tweet
    output: df with new columns
    '''
    df['lower_text'] = df[col].apply(lambda x: str(x).lower())
    df['lemmatized'] = df['lower_text'].apply(lambda x: lemmatizer(x))
    tweets = df.lemmatized.values
    return df, tweets

def create_stop_words():
    ''' create the list of stop words
    input: none
    output: list of stop words 
    '''
    stop_words = list(STOPWORDS)
    stop_words.append('rt')
    stop_words.append('crosstalk')
    return stop_words

def define_clusters(train_vector, clusters = 8):
    ''' fits the kmeans clustering model
    input: text to vectorize
    output: fitted kmeans model
    '''
    kmeans = KMeans(n_clusters = clusters)
    kmeans.fit(train_vector)
    return kmeans

def transform_text(model, text):
    ''' 
    input:
    output:
    '''
    return model.transform(text)

def vectorize(df, column = 'tweet_text'):
    ''' 
    input:
    output:
    '''
    df, tweets = lower_lemma(df, column)
    stop_words = create_stop_words()
    vectorizer, x_train_vect = orig_vect(df[column], stop_words)
    features = vectorizer.get_feature_names()
    vect_features = x_train_vect.toarray().mean(axis = 0)
    top_words = np.array(features)[vect_features.argsort()[::-1][:10]]
    return x_train_vect

def tokenize(text):
    ''' 
    input:
    output:
    ''' 
    stop_words = set(create_stop_words())
    word_tokens = word_tokenize(text)
    word_tokens = [w for w in word_tokens if not w in stop_words]
    word_tokens = lemmatizer(word_tokens)
    return word_tokens
