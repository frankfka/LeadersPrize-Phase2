import time
from typing import List

import gensim


class Word2VecVectorizer(object):
    """
    Loads a word2vec format from a given path
    - binary: whether to load a binary format (ex. GoogleNewsVectors.bin.gz)
    """

    def __init__(self, path):
        start_time = time.time()
        self.model = gensim.models.KeyedVectors.load_word2vec_format(path, unicode_errors='ignore', binary=True)
        print(f"Gensim vectors loaded in {time.time() - start_time}s")

    def get_vectors(self, words: List[str]):
        """
        - Iterate over words, if we use n_grams, construct n-grams and see if vectors exist for them
        - For each word, attempt to get vector from the model
        - Not-In-Vocab -> Not added to returned array
        """
        num_words = len(words)
        vectors = []
        idx = 0

        while idx < num_words:
            # Try tri-grams (in_this_format) if index allows
            if idx + 2 < num_words:
                trigram = words[idx] + '_' + words[idx + 1] + '_' + words[idx + 2]  # ex. new_york_city
                vec = self.__get_word_vec(trigram)
                # If vector is found, append to list, update index, and skip to next iteration of loop
                if vec is not None:
                    vectors.append(vec)
                    idx += 3
                    continue  # Don't consider bi-grams/uni-grams
            # Try bi-grams
            if idx + 1 < num_words:
                bigram = words[idx] + '_' + words[idx + 1]  # ex. donald_trump
                vec = self.__get_word_vec(bigram)
                if vec is not None:
                    vectors.append(vec)
                    idx += 2
                    continue
            # Default to uni-gram
            vec = self.__get_word_vec(words[idx])
            if vec is not None:
                vectors.append(vec)
            idx += 1

        return vectors

    # Word can be a n-gram as well
    def __get_word_vec(self, word):
        # Best possible case - word in model, return that vector
        if word in self.model.vocab:
            return self.model[word]
        # Try a lowercase representation
        if word.lower() in self.model.vocab:
            return self.model[word.lower()]
        # Return none if not found
        return None
