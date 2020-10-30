'''
code to find the top 20 words in the President Debate transcript and create
a column of how many times that tweet contains a word in the top 20
'''
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

def lower_lemma(df, col):
    ''' lower and lemmatize the text 
    input: df = dataframe with text, col = column with text
    output: df with new columns
    '''
    df['lower_text'] = df[col].apply(lambda x: str(x).lower())
    df['lemmatized'] = df['lower_text'].apply(lambda x: lemmatizer(x))
    return df

def orig_vect(X_train, custom_stopwords):
    ''' pass in x training data to fit and transform
    input: text to be vectorized, customer list of stopwords
    output: vectorizer, x_train vectorized
    '''
    vectorizer = TfidfVectorizer(stop_words=custom_stopwords, max_features = 1000, 
                                analyzer='word', ngram_range=(1,2))
    x_train_vect = vectorizer.fit_transform(X_train)
    return vectorizer, x_train_vect

def create_stop_words():
    ''' create the list of stop words
    input: none
    output: list of stop words 
    '''
    stop_words = list(STOPWORDS)
    additional_stop_words = ['rt', 'crosstalk', 'wa', 'realdonaldtrump', 'joebiden', 're', 've']
    for word in additional_stop_words:
        stop_words.append(word)
    return stop_words

def full_run(df, col):
    ''' completes the full run of functions above
    input: datafram with text, column with text
    output: top 20 words from that text
    '''
    new_df = lower_lemma(df, col)
    stop_words = create_stop_words()
    vectorizer, x_train_vect = orig_vect(new_df[0], stop_words)
    features = vectorizer.get_feature_names()
    vect_features = x_train_vect.toarray().mean(axis = 0)
    top_words = np.array(features)[vect_features.argsort()[::-1][:20]]
    return list(top_words) 

if __name__ == "__main__":
    #get top 20 words from transcript
    tran_script = pd.read_json("../notebooks/transcript.json")
    transctip_top20 = full_run(tran_script, 0)

    #import tweets
    df = pd.read_json("../data/basic_dataset.json")
    df = df[['tweet_id', 'tweet_text']].copy()

    #lower and lemmatize tweet
    tweet_df = lower_lemma(df, 'tweet_text')

    #create column to count how many times a word from the debate's top 20 appears in that tweet
    tweet_df['intop20'] = None
    for row, word in enumerate(tweet_df['lemmatized']):
        count = 0
        for w in word.split():
            if w in transctip_top20:
                count += 1
            tweet_df.at[row, 'intop20'] = count
