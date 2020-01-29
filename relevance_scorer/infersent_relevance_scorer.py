from typing import List

from relevance_scorer.infersent_vectorizer import InfersentVectorizer
from util.text_util import cosine_similarity


class InfersentRelevanceScorer:

    def __init__(self, vectorizer: InfersentVectorizer):
        self.vectorizer = vectorizer

    def get_relevance(self, reference_text: str, sentences: List[str]) -> List[float]:
        """
        Infersent works best on batches of sentences, so we take the claim and a list of comparison sentences
        and return a list of similarities
        """
        reference_vector = self.vectorizer.get_sentence_vectors([reference_text])
        assert len(reference_vector) == 1  # We should get 1 vector back
        reference_vector = reference_vector[0]  # Get the vector itself
        # Map sentence vectors to cosine similarity
        return [cosine_similarity(reference_vector, sent_vector) for sent_vector in
                self.vectorizer.get_sentence_vectors(sentences)]
