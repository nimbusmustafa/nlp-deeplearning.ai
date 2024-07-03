import pandas as pd
import zipfile
zf = zipfile.ZipFile('/home/mustafa/nlp/Movie_Review_Sentiment_Analysis/IMDB Dataset.csv.zip') 

data = pd.read_csv(zf.open('IMDB Dataset.csv'))
print(data.head())
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

def preprocess_text(text):
    text = re.sub(r'https?://\S+', '', text)
    
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'<br\s*/?>', ' ', text)

    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    words = text.lower().split()
    
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words if word not in set(stopwords.words('english'))]
    
    return ' '.join(words)

# Assuming 'data' is your DataFrame containing 'review' column
data['review'] = data['review'].apply(preprocess_text)
print(data.head())
data['sentiment'] = data['sentiment'].replace({'positive': 1, 'negative': 0})
def build_freqs(data):
    freqs = {}
    
    for index, row in data.iterrows():
        review = row['review']
        sentiment = row['sentiment']
        words = review.split()
        #print(words,sentiment)
        
        for word in words:
            pair = (word, sentiment)
            if pair not in freqs:
                freqs[pair] = 1
            else:
                freqs[pair] += 1
    
    return freqs
freqs = build_freqs(data)
filtered_freqs={}
for pair, freq in freqs.items():
    if isinstance(pair[1], (int)):
        filtered_freqs[pair] = freq
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.2, random_state=78)

import numpy as np
from collections import defaultdict

def train_naive_bayes(freqs, train_y):
  
    loglikelihood = {}
    logprior = 0

    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    N_pos = N_neg = 0
    for pair, freq in freqs.items():
        try:
            if pair[1] > 0:
                N_pos += freq
            else:
                N_neg += freq
        except TypeError:
            continue

    D = len(train_y)

    D_pos = np.sum(train_y == 1)

    D_neg = np.sum(train_y == 0)

    logprior = np.log(D_pos) - np.log(D_neg)

    for word in vocab:
        freq_pos = freqs.get((word, 1), 0)
        freq_neg = freqs.get((word, 0), 0)

        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        loglikelihood[word] = np.log(p_w_pos / p_w_neg)

    return logprior, loglikelihood

logprior, loglikelihood = train_naive_bayes(filtered_freqs, y_train)

