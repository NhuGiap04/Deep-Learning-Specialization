from sklearn.svm import SVC
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud

# File "Spam.csv" contains some invalid characters
# An error may be thrown, so we need to encode it
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# After encoding, there are some redundant columns
# Drop unnecessary columns
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# rename columns
df.columns = ['labels', 'data']

# create binary labels
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
y = df['b_labels'].values

# try multiple ways of calculating features
# decode_error: ignore any invalid UTF character
tfidf = TfidfVectorizer(decode_error='ignore')
X = tfidf.fit_transform(df['data'])

# count_vectorizer = CountVectorizer(decode_error='ignore')
# X = count_vectorizer.fit_transform(df['data'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model = SVC(kernel='poly', degree=3, C=2.)

t0 = datetime.now()
model.fit(X_train, y_train)
print('train duration:', datetime.now() - t0)

t0 = datetime.now()
print('train score:', model.score(X_train, y_train), 'duration:', datetime.now() - t0)

t0 = datetime.now()
print('test score:', model.score(X_test, y_test), 'duration:', datetime.now() - t0)


# Visualize the data
def visualize(label):
    words = ''
    for msg in df[df['labels'] == label]['data']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(label)
    plt.show()


# visualize('spam')
# visualize('ham')

# see what we're getting wrong
df['predictions'] = model.predict(X)

# things should be spam
print('*** things that should be spam ***')
sneaky_spam = df[(df['predictions'] == 0) & (df['b_labels'] == 1)]['data']
for msg in sneaky_spam:
    print(msg)

# things should not be spam
print('*** things that should not be spam ***')
not_actually_spam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
for msg in not_actually_spam:
    print(msg)

# Test the result
# test_string = input()
# test_string = tfidf.transform([test_string])
#
# print(model.predict(test_string))
