import textwrap

import pandas as pd
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

df = pd.read_csv('../Datasets/bbc_text_cls.csv')
doc = df[df.labels == 'entertainment']['text'].sample(random_state=123)


def wrap(x):
    return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)


summarizer = TextRankSummarizer()
parser = PlaintextParser.from_string(
    doc.iloc[0].split("\n", 1)[1],
    Tokenizer("english"))
summary = summarizer(parser.document, sentences_count=5)

print(summary)

for s in summary:
    print(wrap(str(s)))

summarizer = LsaSummarizer()
summary = summarizer(parser.document, sentences_count=5)
for s in summary:
    print(wrap(str(s)))

# https://radimrehurek.com/gensim_3.8.3/summarization/summariser.html
# https://arxiv.org/abs/1602.03606
# Parameters
# text (str) – Given text.
# ratio (float, optional) – Number between 0 and 1 that determines the
#     proportion of the number of sentences of the original text to be
#     chosen for the summary.
# word_count (int or None, optional) – Determines how many words will the
#     output contain. If both parameters are provided, the ratio will be
#     ignored.
# split (bool, optional) – If True, list of sentences will be returned.
#     Otherwise, joined strings will bwe returned.
# from gensim.summarization.summarizer import summarize
#
# summary = summarize(doc.iloc[0].split("\n", 1)[1])
# print(wrap(summary))
