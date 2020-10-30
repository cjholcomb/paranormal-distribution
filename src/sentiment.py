'''
sentiment analysis class
'''
import pandas as pd 

class VaderSentiment():
    ''' label tweet as negative, neutral, or positive and produce a compound float '''
    def __init__(self):
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()

    def predict(self, df, col):
        ''' return new dataframe with prediction columns on tweet '''
        sentiment_df = pd.DataFrame(columns = ['s_neg','s_neu','s_pos', 'tweet_text', 'tweet_id'])
        tweet_id = df['tweet_id']
        for row, tweet in enumerate(df[col]):
            prob = self.vader.polarity_scores(tweet)
            sentiment_df = sentiment_df.append({'s_neg':prob['neg'], 's_neu':prob['neu'],
                                's_pos':prob['pos'], 'tweet_text':tweet, 
                                'tweet_id':tweet_id[row]}, ignore_index=True)
        return sentiment_df

if __name__ == "__main__":
    pass