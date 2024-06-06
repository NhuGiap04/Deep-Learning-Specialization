from gensim.models import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


def find_analogies(w1, w2, w3):
    # w1 - w2 = ? - w3
    # e.g. king - man = ? - woman
    #      ? = +king +woman -man
    r = word_vectors.most_similar(positive=[w1, w3], negative=[w2])
    print("%s - %s = %s - %s" % (w1, w2, r[0][0], w3))


find_analogies('king', 'man', 'woman')
find_analogies('france', 'paris', 'london')
find_analogies('france', 'paris', 'rome')
find_analogies('paris', 'france', 'italy')
find_analogies('france', 'french', 'english')
find_analogies('japan', 'japanese', 'chinese')
find_analogies('japan', 'japanese', 'italian')
find_analogies('december', 'november', 'june')
find_analogies('miami', 'florida', 'texas')
find_analogies('einstein', 'scientist', 'painter')
find_analogies('man', 'woman', 'she')
find_analogies('man', 'woman', 'aunt')
find_analogies('man', 'woman', 'sister')
find_analogies('man', 'woman', 'wife')
find_analogies('man', 'woman', 'actress')
find_analogies('man', 'woman', 'mother')
find_analogies('nephew', 'niece', 'aunt')


def nearest_neighbors(w):
    r = word_vectors.most_similar(positive=[w])
    print("neighbors of: %s" % w)
    for word, score in r:
        print("\t%s" % word)


nearest_neighbors('king')
nearest_neighbors('france')
nearest_neighbors('japan')
nearest_neighbors('einstein')
nearest_neighbors('woman')
nearest_neighbors('nephew')
nearest_neighbors('february')

# Exercise: download pretrained GloVe vectors from
# https://nlp.stanford.edu/projects/glove/
#
# Implement your own find_analogies() and nearest_neighbors()
#
# Hint: you do NOT have to go hunting around on Stackoverflow
#       you do NOT have to copy and paste code from anywhere
#       look at the file you downloaded
