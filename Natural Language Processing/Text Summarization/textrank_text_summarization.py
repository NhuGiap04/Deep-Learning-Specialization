import pandas as pd
import numpy as np
import textwrap
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('../Datasets/bbc_text_cls.csv')

doc = df[df.labels == 'business']['text'].sample(random_state=42)


def wrap(x):
    return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)


print(wrap(doc.iloc[0]))

print(doc.iloc[0].split("\n", 1)[1])

sents = nltk.sent_tokenize(doc.iloc[0].split("\n", 1)[1])

featurizer = TfidfVectorizer(
    stop_words=stopwords.words('english'),
    norm='l1')

X = featurizer.fit_transform(sents)

# compute similarity matrix
S = cosine_similarity(X)

# normalize similarity matrix
S /= S.sum(axis=1, keepdims=True)

# uniform transition matrix
U = np.ones_like(S) / len(S)

# smoothed similarity matrix
factor = 0.15
S = (1 - factor) * S + factor * U

# find the limiting / stationary distribution
eigenvals, eigenvecs = np.linalg.eig(S.T)
print(eigenvecs[:, 0] / eigenvecs[:, 0].sum())

limiting_dist = np.ones(len(S)) / len(S)
threshold = 1e-8
delta = float('inf')
iters = 0
while delta > threshold:
    iters += 1

    # Markov transition
    p = limiting_dist.dot(S)

    # compute change in limiting distribution
    delta = np.abs(p - limiting_dist).sum()

    # update limiting distribution
    limiting_dist = p

print(iters)
print(limiting_dist)

print(np.abs(eigenvecs[:, 0] / eigenvecs[:, 0].sum() - limiting_dist).sum())

scores = limiting_dist

sort_idx = np.argsort(-scores)

# Many options for how to choose which sentences to include:

# 1) top N sentences
# 2) top N words
# 3) top X% sentences or top X% words
# 4) sentences with scores > average score
# 5) sentences with scores > factor * average score

# You also don't have to sort. May make more sense in order.

print("Generated summary:")
for i in sort_idx[:5]:
    print(wrap("%.2f: %s" % (scores[i], sents[i])))

print(doc.iloc[0].split("\n")[0])


def summarize(text, factor=0.15):
    # extract sentences
    sents = nltk.sent_tokenize(text)

    # perform tf-idf
    featurizer = TfidfVectorizer(
        stop_words=stopwords.words('english'),
        norm='l1')
    X = featurizer.fit_transform(sents)

    # compute similarity matrix
    S = cosine_similarity(X)

    # normalize similarity matrix
    S /= S.sum(axis=1, keepdims=True)

    # uniform transition matrix
    U = np.ones_like(S) / len(S)

    # smoothed similarity matrix
    S = (1 - factor) * S + factor * U

    # find the limiting / stationary distribution
    eigenvals, eigenvecs = np.linalg.eig(S.T)

    # compute scores
    scores = eigenvecs[:, 0] / eigenvecs[:, 0].sum()

    # sort the scores
    sort_idx = np.argsort(-scores)

    # print summary
    for i in sort_idx[:5]:
        print(wrap("%.2f: %s" % (scores[i], sents[i])))


doc = df[df.labels == 'entertainment']['text'].sample(random_state=123)
summarize(doc.iloc[0].split("\n", 1)[1])

print(doc.iloc[0].split("\n")[0])
