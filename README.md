# paranormal-distribution
2020 Election Galvanize Alumni Datathon
<!-----
NEW: Check the "Suppress top comment" option to remove this info from the output.

Conversion time: 1.287 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs to Markdown version 1.0β29
* Fri Oct 30 2020 10:23:24 GMT-0700 (PDT)
* Source doc: Datathon README

WARNING:
You have 2 H1 headings. You may want to use the "H1 -> H2" option to demote all headings by one level.

----->


<p style="color: red; font-weight: bold">>>>>>  gd2md-html alert:  ERRORs: 0; WARNINGs: 1; ALERTS: 0.</p>
<ul style="color: red; font-weight: bold"><li>See top comment block for details on ERRORs and WARNINGs. <li>In the converted Markdown or HTML, search for inline alerts that start with >>>>>  gd2md-html alert:  for specific instances that need correction.</ul>

<p style="color: red; font-weight: bold">Links to alert messages:</p>
<p style="color: red; font-weight: bold">>>>>> PLEASE check and correct alert issues and delete this message and the inline alerts.<hr></p>


No Merge Conflicts!!! 

Notes and braindumb: 



*   Sentiment analysis using clustering - warmer toward biden or trump? 
*   Opponent vs supporter 
*   Is someone tagged vs mentioned
*   Predict likes and retweets
*   **Engagement of tweets. Favorites, replies, retweets, follower-count, verified accounts.**
*   Higher follower count have X topics
*   Compare topics of debate to what is said on twitter
*   “Snitch tags” 
*   User.verified status 
*   Popular on twitter, what do they say about the candidates 
*   Relationship between number of followers and sentiment score (Pro T, Pro B, Neg T, Neg B)
*   Clustering of tweets and possible connection to transcript 
    *   [Solution example from class](https://github.com/GalvanizeDataScience/solutions-RFT4/blob/master/clustering/Clustering_Solutions.ipynb)
*   **Cluster tweets by debate topics. Find both Pro and Neg tweets for both candidates**
*   **Increase twitter profile what do I want to say to get more engagement - how do I do that? What information is getting the most traction **

**FINAL:**

To increase your RT here is what we are finding you should post after a debate



*   Topics from debate transcript 
*   Cosine similarity between transcript clusters and tweet
*   **Sentiment analysis of tweet itself Pos or Neg**
*   Target RT or Likes

Feature importance - how much does having a picture matter

Heather’s notes on finding influence of top ten mentions:



*   Using ‘mentions’ column proved to be very difficult. Each row had a string with a nested list and dictionary in that order. Some of the rows which had multiple mentions had a tuple of dictionaries as well which caused many difficulties when trying to extract the screen name of the profile mentioned. Parsing through the text data, I one hot encoded weather one of the top ten most used mentions were involved to see if we could use those features to predict retweet count. 
*   

**MUST HAVES FOR RUBRIC**

[https://docs.google.com/spreadsheets/d/1EBdSDVZkUUw2p0Km2JVYFqisvQPkW1JVVRoe_OlsI1o/edit#gid=859454411](https://docs.google.com/spreadsheets/d/1EBdSDVZkUUw2p0Km2JVYFqisvQPkW1JVVRoe_OlsI1o/edit#gid=859454411) 



*   Process
    *   Describe each step (get DF, then separated, etc.
    *   List all assumptions
    *   Explain data source **FOLLOW UP
*   Business problem / research question clearly defined
    *   How is it useful?
    *   Successful criteria for completion
*   Code
    *   Explain all folders and files
    *   Can run as is after a git clone
    *   Organized 
    *   PEP8
    *   Docstrings
    *   Remove unnecessary files
    *   Only .py
    *   Model baseline is trained and evaluated


# How to get attention from political twitter


## Background

As part of the Galvanize inaugural Datathon, we were invited to produce a working model and business case using Twitter api data, just prior to the election. The dataset includes three days of tweets from high quality twitter accounts [explanation of this?] from 30 September 2020 to 2 October 2020, three days immediately after the first Presidential Debate.


### Business Case

Political Twitter is a volatile, energetic, engaging mess of a conversation on a good day. Directly after a presidential debate will only make the energy rise further. Additionally,Twitter itself has risen to become one of the most popular social media platforms in the world, and is a critical component of social media presence, as well as personal and professional brand management. news organizations, nonprofits, journalists, comedians, podcasters, artists, actors, political operatives, candidates for office, and academics are all trying to raise their profile by engaging with political content on Twitter, and with this project we will attempt to model how a user might capitalize on political events to get more engagement on their tweets. 



*   Imagine you are new to Twitter and you are trying to build up the engagement on your account. You are really interested in what is happening right now with the 2020 Election (sidenote, who isn’t?) and you want to contribute to the conversation. What is the best way to do that? Who should you be mentioning, retweeting and what words get the most action?


## Data

In addition to the Twitter data provided by Galvanize, we also made use of the first presidential debate transcripts.


### Sources

Instructions for obtaining Twitter data used in this project:



*   Data Source: [GitHub Repo link provided @12PM Friday 10/29/2020]
1. Clone [GitHub Repo link provided @12PM Friday 10/29/2020]
2. Follow the instructions in the repository to install twarc and tqdm.
3. Apply for a twitter developer account.
4. Save api key, save api secret key, save bearer token.
5. Enter your twitter api information into twarc.
6. Use a mv command to move the contents of the desired days into a new single directory.
7. Look inside the cloned repository for the appropriate .txt files containing tweet ids. (ex. cat * >> file name.txt)
8. Concatenate those files into one file.
9. In the terminal, use awk 'NR % 100 == 0' &lt;file.txt> > &lt;result.txt> to systematically sample every 100th tweet id. These are the tweets you will hydrate.
10. Modify the hydrate.py script in the cloned repository and run the script to rehydrate tweets from your file of tweet ids.
11. Analyze tweets.

Instructions for obtaining debate transcript:

The debate transcript can be [found here]([https://www.rev.com/blog/transcripts/donald-trump-joe-biden-1st-presidential-debate-transcript-2020](https://www.rev.com/blog/transcripts/donald-trump-joe-biden-1st-presidential-debate-transcript-2020))

The data was organized manually. To replicate:



1. Copy the full debate transcript into a .txt file.
2. Replace blocks of text indicating a new speaker (including the timestamp) with [‘,’]
3. Add square brackets to the beginning and end of the file.
4. You should now have a list file that can serve as a corpus.

Exploratory Data Analysis

After the initial import of the large data set, we realized that this json file consisted of 140,000 tweets and over 370 columns. Parsing the columns revealed several key categories:



1. Content of the tweet (e.g. full tweet text, user mentions, hashtags)
2. Tweet metadata (e.g. reply/quote/retweet, time)
3. User information (e.g. usernames, profile fields, follower count)
4. Content and metadata of retweeted/quoted tweets

Given we were interested in the effect the debate had on twitter engagement, we decided early on that the features directly from the dataset would be limited to the most relevant, and spend more time working with natural language processing.

Before diving into the NLP, we took a look at some of the features that might affect retweets and favorites. Simply noting the length of a tweet, could we determine if there was any correlation? The visual below highlights that tweets over 304 characters seem to have no correlation to an increase in favorites, this is the start of our interest in predicting the engagement with our twitter data.

SCATTERPLOT HERE FAVORITE/LENGTH OF TEXT

Moving into the actual language of the text, we note the following word cloud highlights the top terms mentioned in the twitter text. 

WORDCLOUD VIS

As our goal was to model how to increase Twitter engagement, we decided to choose two variables as our targets- the number of retweets each tweet received, and the number of favorites (likes) it received, using these as a proxy for Twitter engagement. Disappointingly, number of replies and number of views were not available in this dataset (or anywhere on the Twitter api).

For potential expansion of the project, we also retained the user and tweet metadata associated with tweets quoted and retweeted. If quoting/retweeting proves to have high feature importance, we would retain the ability to drill further down and see what tweets increase engagement when quoted/retweeted.

 **ADD ON TO - VISUAL?

We also wanted to supplement our feature set with some derived features. They come from three sources:

Sentiment analysis of the tweet text.

Identify the top 20 words in the transcript and count how many of those words appeared in the tweet.

Hashtags/user mentions used in the tweet text.

Please see the data dictionary [here](link) for a breakdown of all variables


### Sentiment Analysis Explanation

We knew we wanted to bring in sentiment analysis to this dataset. With many different choices available to use for sentiment analysis we needed to find something that could use our unlabeled text and classify each Tweet as positive, neutral, or negative. We decided to use the [VADER]([[https://github.com/cjhutto/vaderSentiment](https://github.com/cjhutto/vaderSentiment)) library. VADER stands for Valence Aware Dictionary for sEntiment Reasoning (they really forced the E to work there) and is specifically tuned for social media expressions.

What makes VADER special is its ability to use text sentiment and be sensitive to both polarity and intensity of emotion in text. It is able to recognize that “happy” has a different meaning than when it is paired with “not” and it can also understand the emphasis of capitalization in words and differentiate “LOVE” from “love” in its final score of that word.

Each tweet is given a percentage of being “negative” “neutral” or “positive” we identified the highest percentage as the “final label” of the tweets sentiment label and analyzed those results.

[visual]

It is very interesting to see that the majority of our tweets are falling into the “neutral” category. With additional time we would have explored the “compound” number which is a normalized score of the tweet. That may be a better representation of sentiment for our tweet.


### Debate Transcript Topic Mining

To determine the effect of the debate on twitter engagement, we attempted to process the debate transcript and draw out latent topics.

Producing the corpus as described above, we had individual “speeches” (speech between questions/interruptions/allotted time expiring) as 790 documents. We removed stop words and lemmatized the text before tokenizing and vectorizing it, and then used KMeans clustering to derive clusters. Without specifying a number of clusters, the model produced 8, which is what we decided to initially explore.

After deriving the topics, **visual?!** we then attempted to vectorize the tweet text itself. This avenue of inquiry has been put aside for now, as time does not permit us to complete these features before the closing of the Datathon.


### Hashtags/Mentions

Looping through the text data, we were able to gather the top 10 hashtags and mentions used within the three day period after the election. The idea was to one hot encode this information to see if using these top ten hashtags and the top ten tags led to more retweets than those twitter texts that did not mention those top options. The visual below shows the top ten hashtags and the top ten mentions which were features thrown into our final model. 

VISUAL OF TOP TENS HERE


## Model


### Evaluation Metric

As our goal here is to provide guidance for someone looking to increase their Twitter engagement, we decided to use Explained Variance as our model evaluation metric. This does run the risk of our model having higher bias, and we debated additional metrics, but ultimately we value reducing the uncertainty in behavior effects geared toward increasing engagement.


### Baseline Models

We used seven different regressor model types as our baseline (Linear Regression, K Neighbors, Decision Tree, Random Forest, Gradient Boost, Support Vector, and XGBoost.)

[bar chart]

Linear Regression and SVR did not perform well, but all the other 5 had Explained Variance/r2 scores above .7, and three above .8. The winner was XGBoost, which came in above .85.


### Model Tuning

As time is not a plentiful resource for the Datathon, using gridsearch, a highly processor-intensive and time-consuming task. We did do some model tuning on our XGBoost model, and after two rounds of using GridSearchCV, we brought our r2 up to .875. Additional tuning could improve this, however it will need to be undertaken after the Datathon is completed.


### Feature Importance

[correlation matrix]

As shown above, there isn’t much unexpected correlation between our features. Is_reply is mutually exclusive with both is_quote and is_retweet, so a strong negative correlation is expected. Additionally, s_neu measures the absence of s_neg and s_pos, so that strong negative correlation is also expected.

There are some notable lack of correlations as well here. Verified users are public figures and likely to have more followers, but the correlation is unexpectedly low. Additionally, “egg” users (those without profile pictures) are widely regarded as new/naive/tech-unsavvy, so the lack of a correlation here is striking

[feature importance1]

By far the most important feature seems to be s_neu (neutral language), which outstrips all the others. However this is computed as the ‘absence’ of positive/negative sentiment. The high correlation with its counterpart features, coupled with the fact that 98% of all tweets end up in the “neutral” category, suggests this may be hiding other important information.

[feature importance2]

Dropping this feature from the model shows something very different. Here ‘positive’ sentiment is quite high in importance. 


## Suggested Conclusions

While there is much more research to perform in this area, there are some tentative conclusions we can draw from what we have.

_The content of the tweet is more important in driving engagement than the existing profile of the user._

“Control” features (those that the user cannot change easily) like verified, followers, and egg, are surprisingly not that important determinants of tweet engagement (none crack .25 on feature importance).

_The text of the tweet matters more than the metadata_

The high importance of the s_pos feature indicates that it’s the actual text that matters, rather than quoting or retweeting.

IMPORTANT NOTE: The team is not satisfied with the results of the sentiment analysis as far as classifying tweets as positive, negative or neutral. It’s important to read this finding as “Using speech patterns designated Positive in the Vader Sentiment Model might increase engagement” and *not* “Using positive language might increase engagement”

_There really needs to be a specific political social media sentiment library._


## Next Steps



*   We would like to dive deeper into the Sentiment Analysis and tokenize the words found in “positive” tweets vs “negative tweets.” We were curious about the fact that 98% of our tweets were falling into the “neutral” category. Due to the opacity of the Vader library, however, we don’t have much insight into exactly what defines positive/negative/neutral.
*   We would like to complete the cosine difference between the debate transcript and the tweet text, and add those features to our model.
*   Some more time and energy can be spent on hyperparameter tuning our model, and we may want to also tune our Gradient Boost & K Nearest Neighbors models.
*   We would also like to add three days worth of tweets *before* the debate, to measure any shifts and topics/engagement


# Appendix: File Explanations

**VIDEO:**



*   Business problem
*   Sentiment and text vectorization 
*   Features for model
*   Model performance