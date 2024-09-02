import numpy as np
import pandas as pd
import textwrap
import nltk
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

np.random.seed(1234)

# nltk.download('punkt')

df = pd.read_csv('../Datasets/bbc_text_cls.csv')
df.head()

labels = set(df['labels'])
print(labels)

# Pick a label whose data we want to train from
label = 'business'

texts = df[df['labels'] == label]['text']
texts.head()

# collect counts
probs = {}  # key: (w(t-1), w(t+1)), value: {w(t): count(w(t))}

for doc in texts:
    lines = doc.split("\n")
    for line in lines:
        tokens = word_tokenize(line)
        for i in range(len(tokens) - 2):
            t_0 = tokens[i]
            t_1 = tokens[i + 1]
            t_2 = tokens[i + 2]
            key = (t_0, t_2)
            if key not in probs:
                probs[key] = {}

            # add count for middle token
            if t_1 not in probs[key]:
                probs[key][t_1] = 1
            else:
                probs[key][t_1] += 1

# normalize probabilities
for key, d in probs.items():
    # d should represent a distribution
    total = sum(d.values())
    for k, v in d.items():
        d[k] = v / total

detokenizer = TreebankWordDetokenizer()
print(texts.iloc[0].split("\n")[2])
print(detokenizer.detokenize(word_tokenize(texts.iloc[0].split("\n")[2])))




def sample_word(d):
    p0 = np.random.random()
    cumulative = 0
    for t, p in d.items():
        cumulative += p
        if p0 < cumulative:
            return t
    assert (False)  # should never get here


def spin_line(line):
    tokens = word_tokenize(line)
    i = 0
    output = [tokens[0]]
    while i < (len(tokens) - 2):
        t_0 = tokens[i]
        t_1 = tokens[i + 1]
        t_2 = tokens[i + 2]
        key = (t_0, t_2)
        p_dist = probs[key]
        if len(p_dist) > 1 and np.random.random() < 0.3:
            # let's replace the middle word
            middle = sample_word(p_dist)
            output.append(t_1)
            output.append("<" + middle + ">")
            output.append(t_2)

            # we won't replace the 3rd token since the middle
            # token was dependent on it
            # instead, skip ahead 2 steps
            i += 2
        else:
            # we won't replace this middle word
            output.append(t_1)
            i += 1
    # append the final token - only if there was no replacement
    if i == len(tokens) - 2:
        output.append(tokens[-1])
    return detokenizer.detokenize(output)


def spin_document(doc):
    # split the document into lines (paragraphs)
    lines = doc.split("\n")
    output = []
    for line in lines:
        if line:
            new_line = spin_line(line)
        else:
            new_line = line
        output.append(new_line)
    return "\n".join(output)


i = np.random.choice(texts.shape[0])
doc = texts.iloc[i]
new_doc = spin_document(doc)

print(textwrap.fill(
    new_doc, replace_whitespace=False, fix_sentence_endings=True))

# Extension exercises:
# POS tags
# synonyms
# larger context window
