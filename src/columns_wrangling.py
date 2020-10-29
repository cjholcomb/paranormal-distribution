#full list of features to be extracted from original dataset
features_original = ['Unnamed: 0', 'created_at','id',

 'full_text', 
 
 'entities.user_mentions', 'entities.hashtags', 'possibly_sensitive','in_reply_to_status_id', 'is_quote_status', 
 
 'user.followers_count','user.created_at', 'user.verified', 'user.default_profile_image', 
 
 'quoted_status.entities.user_mentions', 'quoted_status.entities.hashtags', 'quoted_status.possibly_sensitive', 'quoted_status.in_reply_to_status_id', 'quoted_status.is_quote_status', 'quoted_status.retweet_count', 'quoted_status.favorite_count', 
 
 'quoted_status.user.created_at', 'quoted_status.user.followers_count', 'quoted_status.user.default_profile_image', 'quoted_status.user.verified',
 
 'retweeted_status.entities.user_mentions', 'retweeted_status.entities.hashtags', 'retweeted_status.possibly_sensitive', 'retweeted_status.in_reply_to_status_id', 'retweeted_status.is_quote_status',  'retweeted_status.retweet_count', 'retweeted_status.favorite_count', 
 
 'retweeted_status.user.created_at', 'retweeted_status.user.followers_count', 'retweeted_status.user.verified', 'retweeted_status.user.default_profile_image',
 
 'retweet_count', 'favorite_count']

features_basic = ['is_reply','is_quote','is_retweet','sensitive','u_verified','u_egg', 'u_followers', 'u_created']

 #dictionary to rename the fields, make them more manageable/accesible
rename_schema = {'Unnamed: 0':'index', 'created_at':'tweet_date', 'id':'tweet_id', 'full_text':'tweet_text', 'is_quote_status':'is_quote', 'retweet_count':'retweets', 'favorite_count':'favorites', 'possibly_sensitive':'sensitive', 'entities.hashtags':'hashtags',
'entities.user_mentions':'mentions', 

'user.followers_count':'u_followers', 'user.created_at':'u_created', 'user.verified':'u_verified', 'user.default_profile_image':'u_egg',

'quoted_status.entities.hashtags':'q_hashtags', 'quoted_status.entities.user_mentions':'q_mentions', 'quoted_status.possibly_sensitive':'q_sensitive', 'quoted_status.retweet_count':'q_retweets', 'quoted_status.favorite_count':'q_favorites', 'quoted_status.is_quote_status':'q_quote', 

'quoted_status.user.followers_count':'qu_followers', 'quoted_status.user.created_at':'qu_created', 'quoted_status.user.verified':'qu_verified', 'quoted_status.user.default_profile_image':'qu_egg', 

'retweeted_status.entities.hashtags':'r_hashtags', 'retweeted_status.entities.user_mentions':'r_mentions', 'retweeted_status.is_quote_status':'r_quote', 'retweeted_status.retweet_count':'r_retweets','retweeted_status.favorite_count':'r_favorites', 'retweeted_status.possibly_sensitive':'r_sensitive', 

'retweeted_status.user.followers_count':'ru_followers', 'retweeted_status.user.created_at':'ru_created', 'retweeted_status.user.verified':'ru_verified', 'retweeted_status.user.default_profile_image':'ru_egg'}

#final column order for the cleaned dataset
column_order = [ 'tweet_id',
 'tweet_date',
 'tweet_text',
 
 'is_reply',
 'is_quote',
 'is_retweet',
 'sensitive',

 'hashtags',
 'mentions',

 'u_verified',
 'u_egg',

 'u_followers',
 'u_created',

 'q_is_reply',
  'q_quote',
  'q_sensitive',

 'q_retweets',
 'q_favorites',

 'q_hashtags',
 'q_mentions',

 'qu_verified',
 'qu_egg',

 'qu_followers',
 'qu_created',

'r_is_reply',
 'r_quote',
  'r_sensitive',

 'r_retweets',
 'r_favorites',

 'r_hashtags',
 'r_mentions',

 'ru_verified',
 'ru_egg',

 'ru_followers',
 'ru_created',

 'retweets',
 'favorites']

 #specific quote-tweet columns
quote_columns = [
 'q_is_reply',
  'q_quote',
  'q_sensitive',

 'q_retweets',
 'q_favorites',

 'q_hashtags',
 'q_mentions',

 'qu_verified',
 'qu_egg',

 'qu_followers',
 'qu_created']

 #specific retweet columns
retweet_columns = ['r_is_reply',
 'r_quote',
  'r_sensitive',

 'r_retweets',
 'r_favorites',

 'r_hashtags',
 'r_mentions',

 'ru_verified',
 'ru_egg',

 'ru_followers',
 'ru_created',]

 #columns to drop for the basic dataset
basic_drop = [

 'q_is_reply',
  'q_quote',
  'q_sensitive',

 'q_retweets',
 'q_favorites',

 'q_hashtags',
 'q_mentions',

 'qu_verified',
 'qu_egg',

 'qu_followers',
 'qu_created',

'r_is_reply',
 'r_quote',
  'r_sensitive',

 'r_retweets',
 'r_favorites',

 'r_hashtags',
 'r_mentions',

 'ru_verified',
 'ru_egg',

 'ru_followers',
 'ru_created',

 'retweets',
 'favorites',]