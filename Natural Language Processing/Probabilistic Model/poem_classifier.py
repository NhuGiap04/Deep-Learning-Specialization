import stringprep

import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

input_files = ['../Datasets/edgar_allan_poe.txt',
               '../Datasets/robert_frost.txt']

# Collect data into lists
input_texts = []
labels = []

for label, f in enumerate(input_files):
    print(f"{f} corresponds to label {label}")

    for line in open(f):
        line = line.rstrip().lower()
        if line:
            # remove punctuation
            line = line.translate(str.maketrans('', '', string.punctuation))

            input_texts.append(line)
            labels.append(label)

train_text, test_text, T_train, T_test = train_test_split(input_texts, labels)
print(len(T_train), len(T_test))

idx = 1
word2idx = {'<unk>': 0}

# Populate word2idx
for text in train_text:
    tokens = text.split()
    for token in tokens:
        if token not in word2idx:
            word2idx[token] = idx
            idx += 1

print(len(word2idx))

# Convert data into integer format
train_text_int = []
test_text_int = []

for text in train_text:
    tokens = text.split()
    line_as_int = [word2idx[token] for token in tokens]
    train_text_int.append(line_as_int)

for text in test_text:
    tokens = text.split()
    line_as_int = [word2idx.get(token, 0) for token in tokens]
    test_text_int.append(line_as_int)

print(train_text_int[100:105])

# Initialize A and pi matrices - for both classes
V = len(word2idx)

A0 = np.ones((V, V))  # Add 1 smoothing
pi0 = np.ones(V)

A1 = np.ones((V, V))  # Add 1 smoothing
pi1 = np.ones(V)


# Compute counts for A and pi
def compute_counts(text_as_int, A, pi):
    for tokens in text_as_int:
        last_idx = None
        for idx in tokens:
            if last_idx is None:
                # it's the first word in a sentence
                pi[idx] += 1
            else:
                # the last word exists, so count a transition
                A[last_idx, idx] += 1

            # Update last idx
            last_idx = idx


compute_counts([t for t, y in zip(train_text_int, T_train) if y == 0], A0, pi0)
compute_counts([t for t, y in zip(train_text_int, T_train) if y == 1], A1, pi1)

# Normalize A and pi so that they are valid probabilities matrices
A0 /= A0.sum(axis=1, keepdims=True)
pi0 /= pi0.sum()

A1 /= A1.sum(axis=1, keepdims=True)
pi1 /= pi1.sum()

# Log A and pi since we don't need actual probs
logA0 = np.log(A0)
logpi0 = np.log(pi0)

logA1 = np.log(A1)
logpi1 = np.log(pi1)

# Compute priors
count0 = sum(y == 0 for y in T_train)
count1 = sum(y == 1 for y in T_train)
total = len(T_train)
p0 = count0 / total
p1 = count1 / total
logp0 = np.log(p0)
logp1 = np.log(p1)

print(p0, p1)


# Build a Classifier
class Classifier:
    def __init__(self, log_As, log_pis, log_priors):
        self.log_As = log_As
        self.log_pis = log_pis
        self.log_priors = log_priors
        self.K = len(log_priors)  # Number of classes

    def _compute_log_likelihood(self, input_, class_):
        logA = self.log_As[class_]
        logpi = self.log_pis[class_]

        last_idx = None
        logprob = 0
        for idx in input_:
            if last_idx is None:
                # it's the first token
                logprob += logpi[idx]
            else:
                logprob += logA[last_idx, idx]

            # Update last_idx
            last_idx = idx

        return logprob

    def predict(self, inputs):
        predictions = np.zeros(len(inputs))
        for i, input_ in enumerate(inputs):
            posteriors = [self._compute_log_likelihood(input_, c) + self.log_priors[c] for c in range(self.K)]
            pred = np.argmax(posteriors)
            predictions[i] = pred
        return predictions


# Each array must be in order since classes are assumed to index these lists
clf = Classifier([logA0, logA1], [logpi0, logpi1], [logp0, logp1])

P_train = clf.predict(train_text_int)
print(f"Train Acc: {np.mean(P_train == T_train)}")

P_test = clf.predict(test_text_int)
print(f"Test Acc: {np.mean(P_test == T_test)}")

cm = confusion_matrix(T_train, P_train)
print("Train Confusion Matrix")
print(cm)

cm_test = confusion_matrix(T_test, P_test)
print("Test Confusion Matrix")
print(cm_test)

print(f"F1 Train Score: {f1_score(T_train, P_train)}")
print(f"F1 Test Score: {f1_score(T_test, P_test)}")
