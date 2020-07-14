import spacy

from preprocess import text_util


class SpacyRelevanceScorer:

    def __init__(self, spacy_model):
        self.spacy_model = spacy_model

    def get_relevance(self, text: str, other_text: str, max_seq_len: int = 150) -> float:
        """
        Gets the cosine similarity between the two lists of tokens, truncating each to max_seq_len
        This is the most general case and ignores the presence of sentences, so we can compare any length/sequence
        of text.
        """
        return self.spacy_model(text).similarity(self.spacy_model(other_text))
