''' 
molly's py file 
'''
import pandas as pd 
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

def setup_df(path):
    '''
    set up dataframe of tweets
    input: 
        path to jsonl file
    output: 
        dataframe of all information
    '''
    with open('data/concatenated_abridged.jsonl', 'r') as json_file:
        json_lines = json_file.read().splitlines()
    df_inter = pd.DataFrame(json_lines)
    df_inter.columns = ['json_element']
    df_inter['json_element'].apply(json.loads)
    df_final = pd.json_normalize(df_inter['json_element'].apply(json.loads))
    df_final.to_json('full_tweets.json')

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
    pass

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