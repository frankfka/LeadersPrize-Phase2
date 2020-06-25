import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import data_processing
import parameters
import re
from nltk.corpus import stopwords
import pandas as pd
import scipy
import scipy.spatial
import nltk

import vocab

parameters = parameters.parameters
# nltk.download('stopwords')

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


def cosine_distance_between_two_words(word1, word2):
    return 1 - scipy.spatial.distance.cosine(model[word1], model[word2])


def calculate_heat_matrix_for_two_sentences(s1, s2):
    s1 = preprocess(s1)
    s2 = preprocess(s2)
    result_list = [[cosine_distance_between_two_words(word1, word2) for word2 in s2] for word1 in s1]
    result_df = pd.DataFrame(result_list)
    result_df.columns = s2
    result_df.index = s1
    return result_df


def cosine_distance(s1, s2, embeddings=None, word_indices=None):
    if word_indices is None:
        word_indices = vocab.load_dictionary()
    if embeddings is None:
        embeddings = data_processing.loadEmbedding_rand(parameters["embedding_data_path"], word_indices)

    vector_1 = np.mean([embeddings[word_indices[word]] for word in preprocess(s1, word_indices=word_indices)], axis=0)
    vector_2 = np.mean([embeddings[word_indices[word]] for word in preprocess(s2, word_indices=word_indices)], axis=0)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    return cosine


if __name__ == '__main__':
    statements = [
        "The cow is on the grass",
        "The cow stands in the field",
        "There is a cow in the grass",
        "In the grass, there's a cow",
        "The cow is in the city",
        "The cow is nowhere near the grass",
        "There's no cow in the field",
        "The grass field is empty",
        "This is totally irrelevant",
        "Bananas are rich in potassium",
        'The president greets the press in Chicago',
        'Obama speaks to the media in Illinois'
    ]

    model = loadGloveModel(gloveFile)

    for statement_a in statements:
        for statement_b in statements:
            distance = cosine_distance(statement_a, statement_b)
            print(f'{statement_a} vs {statement_b} similarity: {1 - distance}')
    print('debug')
