from typing import List

import numpy as np

from analyze.sentence_relevance_scorer.word2vec_vectorizer import Word2VecVectorizer
from preprocess import text_util
from preprocess.text_util import cos_sim


class Word2VecRelevanceScorer:

    def __init__(self, vectorizer: Word2VecVectorizer):
        self.vectorizer = vectorizer

    def get_relevance(self, text: str, other_text: str, max_seq_len: int = 150) -> float:
        """
        Gets the cosine similarity between the two lists of tokens, truncating each to max_seq_len
        This is the most general case and ignores the presence of sentences, so we can compare any length/sequence
        of text.
        """
        tokens = text_util.tokenize_by_word(text)
        other_tokens = text_util.tokenize_by_word(other_text)
        if len(tokens) > max_seq_len:
            tokens = tokens[0:max_seq_len]
        if len(other_tokens) > max_seq_len:
            other_tokens = other_tokens[0:max_seq_len]
        vectors = self.vectorizer.get_vectors(tokens)
        other_vectors = self.vectorizer.get_vectors(other_tokens)
        return self.__cos_sim(vectors, other_vectors)

    # Returns cosine similarity between two texts
    def __cos_sim(self, vectors: List[str], other_vectors: List[str]) -> float:
        if len(vectors) == 0 or len(other_vectors) == 0:
            return 0
        # Get average vectors of the texts
        avg_vec = self.__get_avg_vec(vectors).reshape(1, -1)  # Reshape to get a 2D array of a single sample
        avg_other_vec = self.__get_avg_vec(other_vectors).reshape(1, -1)
        return cos_sim(avg_vec, avg_other_vec)

    # Returns a column vector resulting from taking an element-wise mean
    def __get_avg_vec(self, vectors):
        return np.mean(vectors, axis=0)