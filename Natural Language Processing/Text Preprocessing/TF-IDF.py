import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize


# nltk.download('punkt')
df = pd.read_csv('../Datasets/bbc_text_cls.csv')
print(df.head())


# Populate word-to-index
# Convert documents into sequences of ints/ids/indices
idx = 0
word2idx = {}
tokenized_docs = []
for doc in df['text']:
    words = word_tokenize(doc.lower())
    doc_as_int = []
    for word in words:
        if word not in word2idx:
            word2idx[word] = idx
            idx += 1
        # Save for later
        doc_as_int.append(word2idx[word])
    tokenized_docs.append(doc_as_int)

# reverse mapping
# if you do it smarter you can store it as a list
idx2word = {v: k for k, v in word2idx.items()}

# number of documents
N = len(df['text'])
# number of words
V = len(word2idx)

# instantiate term-frequency matrix
# note: could have also used count vectorizer
# can use scipy to implement sparse matrix for more efficient memory usage
# dense to sparse
''' -------- Example for Sparse Matrix --------
from numpy import array
from scipy.sparse import csr_matrix
# create dense matrix
A = array([[1, 0, 0, 1, 0, 0], [0, 0, 2, 0, 0, 1], [0, 0, 0, 2, 0, 0]])
print(A)
# convert to sparse matrix (CSR method)
S = csr_matrix(A)
print(S)
# reconstruct dense matrix
B = S.todense()
print(B)'''
tf = np.zeros((N, V))

# populate term-frequency matrix
for i, doc_as_int in enumerate(tokenized_docs):
    for j in doc_as_int:
        tf[i, j] += 1

# Compute IDF
document_freq = np.sum(tf > 0, axis=0)  # document frequency (shape = (V, ))
idf = np.log(N / document_freq)

# compute TF-IDF
# Numpy arrays use Broadcast to compute the product of tf(N x V matrix) and idf(v-length vector)
tf_idf = tf * idf

np.random.seed(123)
# pick a random document, show the top 5
i = np.random.choice(N)
row = df.iloc[i]
print("Label:", row['labels'])
print("Text:", row['text'].split("\n", 1)[0])
print("Top 5 terms:")

scores = tf_idf[i]
indices = (-scores).argsort()

for j in indices[:5]:
    print(idx2word[j])

# Exercises: use CountVectorizer to form the counts instead

# Exercise (hard): use Scipy's csr_matrix instead
# You cannot use X[i, j] += 1 here
