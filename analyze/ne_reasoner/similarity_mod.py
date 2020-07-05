import numpy as np

import re
from nltk.corpus import stopwords
import scipy
import scipy.spatial


def preprocess(raw_text, word_indices):
    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("english"))
    cleaned_words = list(set([w for w in words if w not in stopword_set]))

    filtered_words = [word for word in cleaned_words if word in word_indices.keys()]

    return filtered_words


def cosine_distance(s1, s2, embeddings, word_indices):
    vector_1 = np.mean([embeddings[word_indices[word]] for word in preprocess(s1, word_indices=word_indices)], axis=0)
    vector_2 = np.mean([embeddings[word_indices[word]] for word in preprocess(s2, word_indices=word_indices)], axis=0)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    return cosine
