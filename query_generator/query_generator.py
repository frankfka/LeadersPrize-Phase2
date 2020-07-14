import re

import preprocess.text_util as text_util
from core.models import PipelineClaim


class QueryGenerator:
    """
    Generates a query based on all words in claim + claimant, stripping punctuation and stopwords
    """

    def __init__(self, spacy_model):
        # TODO need to specify in dockerfile
        self.spacy_model = spacy_model

    def get_query(self, claim: PipelineClaim, custom_query: str = "") -> str:
        """
        Generate a query given a claim, the claim should not be preprocessed
        """
        # Query is made of 3 components - quoted strings, the original claim + claimant, and the preprocessed claim
        query = claim.original_claim.claim + ' ' + claim.original_claim.claimant
        query += ' ' + self.__get_quoted_strs(claim.original_claim.claim)
        # query += ' ' + self.__get_entities(claim.original_claim.claim)
        query += ' ' + claim.preprocessed_claim
        return self.__clean(query)

    def __clean(self, text: str) -> str:
        """
        Cleans text given by the claim
        """
        tokens = text_util.tokenize_by_word(text)
        tokens = text_util.clean_tokenized(tokens, lowercase=True, remove_punctuation=True, remove_stopwords=True)
        # Remove short words
        # Note: can consider removing numbers, as they match HTML elements " not tok.isnumeric() and "
        tokens_cleaned = [tok for tok in tokens if len(tok) > 2]
        if tokens_cleaned:
            return ' '.join(tokens_cleaned)
        return ' '.join(tokens)

    def __get_quoted_strs(self, text: str) -> str:
        """
        Get a list of quotes from the claim, we want to preserve these as is in case we match the quote directly
        """
        # Strip quotes
        return ' '.join(re.findall(r'\"(.+)\"', text))

    def __get_entities(self, original_claim: str) -> str:
        spacy_doc = self.spacy_model(original_claim)
        return " ".join(map(lambda entity: entity.text, spacy_doc.ents))
