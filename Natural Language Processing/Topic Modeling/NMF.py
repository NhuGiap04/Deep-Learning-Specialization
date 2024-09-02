import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import nltk
import textwrap

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF

stops = set(stopwords.words('english'))

stops = stops.union({
    'said', 'would', 'could', 'told', 'also', 'one', 'two',
    'mr', 'new', 'year',
})
stops = list(stops)  # required for later version of CountVectorizer

df = pd.read_csv('../Datasets/bbc_text_cls.csv')

vectorizer = CountVectorizer(stop_words=stops)
X = vectorizer.fit_transform(df['text'])

# Note: you could potentially split the data into train and test
# and evaluate the model using the log-likelihood or perplexity
# on out-of-sample data

nmf = NMF(
    n_components=10,  # default: 10
    beta_loss="kullback-leibler",
    solver='mu',
    # alpha_W=0.1,
    # alpha_H=0.1,
    # l1_ratio=0.5,
    random_state=0,
)

nmf.fit(X)


def plot_top_words(model, feature_names, n_top_words=10):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1: -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle('NMF', fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


feature_names = vectorizer.get_feature_names_out()
plot_top_words(nmf, feature_names)

Z = nmf.transform(X)

# Pick a random document
# Check which "topics" are associated with it
# Are they related to the true label?

np.random.seed(0)
i = np.random.choice(len(df))
z = Z[i]
topics = np.arange(10) + 1

fig, ax = plt.subplots()
ax.barh(topics, z)
ax.set_yticks(topics)
ax.set_title('True label: %s' % df.iloc[i]['labels'])


def wrap(x):
    return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)


print(wrap(df.iloc[i]['text']))
