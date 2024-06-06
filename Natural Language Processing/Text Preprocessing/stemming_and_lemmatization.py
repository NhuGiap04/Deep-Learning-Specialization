# Import the libraries
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet

nltk.download("wordnet")
# Get the model Stemming class
porter = PorterStemmer()

print('Stemming Words:')
print(porter.stem("walking"))
print(porter.stem("walked"))
print(porter.stem("walks"))
print(porter.stem("ran"))
print(porter.stem("running"))
print(porter.stem("bosses"))
print(porter.stem("replacement"))

sentence = "Lemmatization is more sophisticated than stemming".split()
for token in sentence:
    print(porter.stem(token), end=" ")

print(porter.stem("unnecessary"))
print(porter.stem("berry"))

# Get the model Lemmatization class
lemmatizer = WordNetLemmatizer()

print('Lemmatization Words:')
print(lemmatizer.lemmatize("walking"))
print(lemmatizer.lemmatize("walking", pos=wordnet.VERB))
print(lemmatizer.lemmatize("going"))
print(lemmatizer.lemmatize("going", pos=wordnet.VERB))
print(lemmatizer.lemmatize("ran", pos=wordnet.VERB), end='\n')

print('Comparing Stemming vs Lemmatization:')
print(porter.stem("mice"))
print(lemmatizer.lemmatize("mice"), end='\n')

print(porter.stem("was"))
print(lemmatizer.lemmatize("was", pos=wordnet.VERB), end='\n')

print(porter.stem("is"))
print(lemmatizer.lemmatize("is", pos=wordnet.VERB), end='\n')

print(porter.stem("better"))
print(lemmatizer.lemmatize("better", pos=wordnet.ADJ), end='\n')


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


print("Lemmatization with Tags:")
nltk.download('averaged_perceptron_tagger')

sentence = "Donald Trump has a devoted following".split()
words_and_tags = nltk.pos_tag(sentence)
print(words_and_tags)

for word, tag in words_and_tags:
    lemma = lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
    print(lemma, end=" ")

sentence = "The cat was following the bird as it flew by".split()
words_and_tags = nltk.pos_tag(sentence)
print(words_and_tags)

for word, tag in words_and_tags:
    lemma = lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
    print(lemma, end=" ")
