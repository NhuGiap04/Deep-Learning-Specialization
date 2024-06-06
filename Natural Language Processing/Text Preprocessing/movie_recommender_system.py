# Import the libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


# Load the dataset
dataset = pd.read_csv('../Datasets/tmdb_5000_movies.csv')
# print(dataset.info())
dataset = dataset.drop(["homepage", "budget", "homepage", "id", "release_date", "vote_average", "vote_count",
                       "popularity", "revenue", "runtime"], axis=1)
# print(dataset.info())
X_initial = dataset.iloc[:, :].values
y_initial = dataset['title'].iloc[:].values


def combine_to_string(initial_X):
    if initial_X.ndim != 2:
        return "Can't convert"
    X = []
    num_of_rows, num_of_columns = initial_X.shape
    for i in range(num_of_rows):
        combined_string = ""
        for j in range(num_of_columns):
            s = str(initial_X[i][j])
            s = re.sub(r'[^A-Za-z0-9 ]+', '', s)
            combined_string += s
            combined_string += " "
        X.append([combined_string])
    return X


X = np.array(combine_to_string(X_initial))
X = np.squeeze(X)
y = np.array(y_initial)

# Start applying TF-IDF
tfidf = TfidfVectorizer(max_features=2000)
X_train = tfidf.fit_transform(X)
print("Shape:", X_train.shape, "Dimension:", X_train.ndim)

# Get User Movie query
query = 'Scream 3'
index = -1
if query in y:
    index = np.where(y == query)[0][0]
    print(index)
else:
    print('No such movies are found')

# Compute similarity between the query Movies and all other movies vectors
if index != -1:
    scores = cosine_similarity(X_train[index], X_train)
    scores = scores.flatten()  # Convert scores to 1D array
    # Get top 5 movies exclude self
    recommend_index = (-scores).argsort()[1: 6]
    for i in recommend_index:
        print(y[i])

