import pandas as pd
from datetime import date
import json
import csv 
import numpy as np
from columns_wrangling import *

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
    # df_inter['json_element'].apply(json.loads)
    df_final = pd.json_normalize(df_inter['json_element'].apply(json.loads))
    return df_final

def dataset_cleanup(df):
    '''
    reorganizes the dataset to be more manageable and understandable

    Arguments:
    df -- Pandas Dataframe (imported from full dataset)'''
    
    #renames columns
    df.rename(columns = rename_schema, inplace = True)

    ## BEGIN BASIC DATASET VARIABLES##

    #creates an "is_reply" 1/0, drops the more complicated one
    df['is_reply'] = df['in_reply_to_status_id'] / df['in_reply_to_status_id']
    df['is_reply'].fillna(value = 0, inplace = True)
    df.drop(columns = ['in_reply_to_status_id'], inplace = True)

    #converts 'is_quote' to 1/0
    df['is_quote'] = df['is_quote'] * 1

    #computes a 1/0 variable for retweets
    df['is_retweet'] = (df['r_retweets'] % 1) + 1
    df['is_retweet'].fillna(0, inplace = True)

    #converts 'U-verified' to 1/0
    df['u_verified'] = df['u_verified'] * 1

    #converts 'sensitive' to 1/0, fills NaNs with 0
    df['sensitive'] = df['sensitive'] * 1
    df['sensitive'].fillna(value = 0, inplace = True)

    #converts egg(default profile picture) to 0/1
    df['u_egg'] = (df['u_egg'] * 1) 

    ## END BASIC DATASET VARIABLES##

    ## BEGIN QUOTE-TWEET DATASET VARIABLES##

    #creates mask for quote-tweets
    q_mask = df['is_quote'] == 1

    #creates a "q_is_reply" 1/0, drops the more complicated one
    df['q_is_reply'] = df['quoted_status.in_reply_to_status_id'] / df['quoted_status.in_reply_to_status_id']
    df['q_is_reply'] = np.where(q_mask, df['q_is_reply'].fillna(0), df['q_is_reply'])
    df.drop(columns = ['quoted_status.in_reply_to_status_id'], inplace = True)

    #converts 'q_quote' to 1/0
    df['q_quote'] = df['q_quote'] * 1

    #converts 'qu_verified' to 1/0
    df['qu_verified'] = df['qu_verified'] * 1

    #converts 'q_sensitive' to 1/0, fills NaNs with 0
    df['q_sensitive'] = df['q_sensitive'] * 1
    df['q_sensitive'] = np.where(q_mask, df['q_is_reply'].fillna(0), df['q_is_reply'])

    #converts egg(default profile picture) to 0/1
    df['qu_egg'] = (df['qu_egg'] * 1) 

    ## END QUOTE-TWEET DATASET VARIABLES##
    
    ## BEGIN RETWEET DATASET VARIABLES##

    #creates mask for quote-tweets
    r_mask = df['is_retweet'] == 1

    #creates a "r_is_reply" 1/0, drops the more complicated one
    df['r_is_reply'] = df['retweeted_status.in_reply_to_status_id'] / df['retweeted_status.in_reply_to_status_id']
    df['r_is_reply'] = np.where(r_mask, df['r_is_reply'].fillna(0), df['r_is_reply'])
    df.drop(columns = ['retweeted_status.in_reply_to_status_id'], inplace = True)

    #converts 'r_quote' to 1/0
    df['r_quote'] = df['r_quote'] * 1

    #converts 'qu_verified' to 1/0
    df['ru_verified'] = df['ru_verified'] * 1

    #converts 'r_sensitive' to 1/0, fills NaNs with 0
    df['r_sensitive'] = df['r_sensitive'] * 1
    df['r_sensitive'] = np.where(r_mask, df['r_is_reply'].fillna(0), df['r_is_reply'])

    #converts egg(default profile picture) to 0/1
    df['ru_egg'] = (df['ru_egg'] * 1)

    ## END RETWEET DATASET VARIABLES##
    
    #reorders columns to group similar features together, in ascending complexity. Targets at the end.
    df = df[column_order]
    
    return df

def basic_dataset(df):
    #drops columns for a very basic dataset. No quote tweet or retweet info.
    return df.drop(columns = basic_drop)

def quote_dataset(df):
    #drops all non-quote-tweet rows, and all retweet variables. Use his dataframe for expanding on quote tweet importance.
    df = df.drop(columns = retweet_columns)
    df = df[df['is_quote'] == 1]
    return df

def retweet_dataset(df):
    #drops all non-retweet rows, and all retweet variables. Use his dataframe for expanding on quote tweet importance.
    df = df.drop(columns = quote_columns)
    df = df[df['is_retweet'] == 1]
    return df

def export_jsons():
    df_basic.to_json('../data/basic_dataset.json')
    df_quote.to_json('../data/quote_dataset.json')
    df_retweet.to_json('../data/retweet_dataset.json')   

if __name__ == '__main__':
    #placeholder df import
    # df = pd.read_csv('../../data/smtweetdata.csv', usecols= features_original)
    path = '../data/concatenated_abridged.jsonl'
    df = setup_df(path)
    df = dataset_cleanup(df)
    df_basic = basic_dataset(df)
    df_quote = quote_dataset(df)
    df_retweet = retweet_dataset(df)
    export_jsons()