''' 
molly's py file 
'''
import pandas as pd 
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
from sklearn.metrics.pairwise import cosine_similarity
from sentiment import *

def setup_df(path):
    '''
    set up dataframe of tweets
    input: 
        path to jsonl file
    output: 
        dataframe of all information
    '''
    with open(path, 'r') as json_file:
        json_lines = json_file.read().splitlines()
    df_inter = pd.DataFrame(json_lines)
    df_inter.columns = ['json_element']
    df_final = pd.json_normalize(df_inter['json_element'].apply(json.loads))
    return df_final

def get_cos_sim(df, col1, col2):
    ''' 
    get the cosine similarity between two vectors in a dataframe 
    input: 
        df = dataframe of tweet information
        col1 = column name the be used in cosine similarity
        col2 = column name to be used in cosine similarity
        
    output:
        array = cosine similairty between two vectors
    '''
    col1 = df[col1].values.reshape(1, -1)
    col2 = df[col2].values.reshape(1, -1)
    return cosine_similarity(col1, col2)

if __name__ == "__main__":
    df = pd.read_json('../data/basic_dataset.json')

    sent = VaderSentiment()
    part = df.sample(n = 5000)
    # BE READY
    sent_df = sent.predict(part, 'tweet_text')

    plt.hist(sent_df['compound'])
    plt.show();

    # for row in range(sent_df.shape[0]):
    #     row_idx = np.argmax(np.array(sent_df.iloc[row,0:3]))
    #     sent_df.at[row, 'compound_cohort'] = row_idx

    for row, num in enumerate(sent_df['compound']):
        if num < -0.5:
            sent_df.at[row, 'compound_category'] = -1
        elif num < 0.5:
            sent_df.at[row, 'compound_category'] = 0
        else:
            sent_df.at[row, 'compound_category'] = 1
    
    plt.hist(sent_df['compound_category'])
    plt.show();

    neg_df = sent_df[sent_df['compound_category'] == -1].copy()
    neu_df = sent_df[sent_df['compound_category'] == 0].copy()
    pos_df = sent_df[sent_df['compound_category'] == 1].copy()

'''
    #EDA
    #verified
    np.sum(df_final['user.verified']) #987
    verified = df_final[['user.verified', 'full_text', 
                        'user.name', 'user.screen_name', 'user.followers_count']].copy()
    verified = verified[verified['user.verified'] == True]

    #followers
    plt.hist(verified['user.followers_count']) 
    plt.show;
'''