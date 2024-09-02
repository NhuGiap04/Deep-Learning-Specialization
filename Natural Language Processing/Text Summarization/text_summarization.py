import pandas as pd
import numpy as np
import textwrap
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('stopwords')

df = pd.read_csv('../Datasets/bbc_text_cls.csv')

print(df.head())

doc = df[df.labels == 'business']['text'].sample(random_state=42)


def wrap(x):
    return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)


sents = nltk.sent_tokenize(doc.iloc[0].split("\n", 1)[1])

featurizer = TfidfVectorizer(
    stop_words=stopwords.words('english'),
    norm='l1',
)

X = featurizer.fit_transform(sents)


def get_sentence_score(tfidf_row):
    # return the average of the non-zero values
    # of the tf-idf vector representation of a sentence
    x = tfidf_row[tfidf_row != 0]
    return x.mean()


scores = np.zeros(len(sents))
for i in range(len(sents)):
    score = get_sentence_score(X[i, :])
    scores[i] = score

# Exercise: can you think of a way to compute all the scores in one step?
sort_idx = np.argsort(-scores)

# Many options for how to choose which sentences to include:

# 1) top N sentences
# 2) top N words or characters.
# 3) top X% sentences or top X% words
# 4) sentences with scores > average score
# 5) sentences with scores > factor * average score

# You also don't have to sort. May make more sense in order.

print("Generated summary:")
for i in sort_idx[:5]:
    print(wrap("%.2f: %s" % (scores[i], sents[i])))

print(doc.iloc[0].split("\n", 1)[0])


def summarize(text):
    # extract sentences
    sents = nltk.sent_tokenize(text)

    # perform tf-idf
    X = featurizer.fit_transform(sents)

    # compute scores for each sentence
    scores = np.zeros(len(sents))
    for i in range(len(sents)):
        score = get_sentence_score(X[i, :])
        scores[i] = score

    # sort the scores
    sort_idx = np.argsort(-scores)

    # print summary
    for i in sort_idx[:5]:
        print(wrap("%.2f: %s" % (scores[i], sents[i])))


doc = df[df.labels == 'entertainment']['text'].sample(random_state=123)
summarize(doc.iloc[0].split("\n", 1)[1])

print(wrap(doc.iloc[0]))
