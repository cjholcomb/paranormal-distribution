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

def lemmatizer(string):
    ''' Lemmatize a string and return it in its original format '''
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w)
                    for w in w_tokenizer.tokenize(string)
                    if "http" not in w])

def orig_vect(X_train, custom_stopwords):
    '''
    pass in x training data to fit and transform to

    Returns:
     vectorizer, x_train vectorized
    '''
    vectorizer = TfidfVectorizer(stop_words=custom_stopwords, max_features = 100, analyzer='word', ngram_range=(1,2))
    x_train_vect = vectorizer.fit_transform(X_train)
    return vectorizer, x_train_vect

def lower_lemma(df, col):
    ''' 
    lower and lemmatize the tweet 
    input: 
        df = dataframe with tweet
        col = column with tweet
    return df with new columns
    '''
    df['lower_text'] = df[col].apply(lambda x: str(x).lower())
    df['lemmatized'] = df['lower_text'].apply(lambda x: lemmatizer(x))
    tweets = df.lemmatized.values
    return df, tweets

def create_stop_words():
    stop_words = list(STOPWORDS)
    stop_words.append('rt')
    return stop_words

if __name__ == "__main__":
    df, tweets = lower_lemma(pos_df, 'tweet')
    stop_words = create_stop_words()
    vectorizer, x_train_vect = orig_vect(tweets, stop_words)
    features = vectorizer.get_feature_names()
    vect_features = x_train_vect.toarray().mean(axis = 0)
    top_words = np.array(features)[vect_features.argsort()[::-1][:10]]
    print(top_words)