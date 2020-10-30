from text_vector import *

transcript = pd.read_json('data/transcript.json')
df = pd.read_json('..data/basic_dataset.json')

train_vector = vectorize(transcript, 0)
clusters = define_clusters(train_vector)

df['tweet_vectorized'] = 