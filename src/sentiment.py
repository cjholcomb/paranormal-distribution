'''
sentiment analysis class
'''
import pandas as pd 

class VaderSentiment():
    ''' Label tweet as negative, neutral, or positive and produce a compound float '''
    def __init__(self):
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()

    def predict(self, df, col):
        ''' return new dataframe with prediction columns on tweet '''
        sentiment_df = pd.DataFrame(columns = ['neg','neu','pos','compound', 'tweet', 'highest_label'])
        for tweet in df[col]:
            prob = self.vader.polarity_scores(tweet)
            sentiment_df = sentiment_df.append({'neg':prob['neg'], 'neu':prob['neu'],
                                'pos':prob['pos'],'compound':prob['compound'],
                                'tweet':tweet}, ignore_index=True)
        return sentiment_df

if __name__ == "__main__":
    pass

    # nltk.downloader.download('vader_lexicon')
    # vader = SentimentIntensityAnalyzer() 


    # df = pd.DataFrame(columns = ['neg','neu','pos','compound', 'tweet'])
    # for t in part['full_text']:
    #     prob = vader.polarity_scores(t)
    #     df = df.append({'neg':prob['neg'], 'neu':prob['neu'],
    #                     'pos':prob['pos'],'compound':prob['compound'],
    #                     'tweet':t}, ignore_index=True)

    # #finding max and min
    # np.where(df['pos'] == np.max(df['pos']))
    # np.where(df['neg'] == np.max(df['neg']))