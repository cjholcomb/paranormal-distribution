import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from src.columns_wrangling import *
import seaborn as sns

sns.set_style('darkgrid')

retweets = {'variable':'retweets', 'kde':False, 'bins':100 , 'color':'blue', 'label':None, 'title':'Retweets', 'ylabel':'Occurance', 'xlabel':'Number of Retweets'}


def histogram(df, variables):
    sns.set_style('darkgrid')
    for key, value in variables.items():
        fig, ax = plt.subplots()
        ax = sns.distplot(df[value['variable']], kde = value['kde'], color = value['color'])
        ax.set_title(value['title'])
        ax.set_ylabel(value['ylabel'])
        ax.set_xlabel(value['xlabel'])
        plt.show();


if __name__ == '__main__':
    df = pd.read_json('../data/basic_dataset.json')