import re
from typing import List

import preprocess.text_util as text_util
from core.models import PipelineClaim


class QueryGenerator:
    """
    Generates a query based on all words in claim + claimant, stripping punctuation and stopwords
    """

    def get_query(self, claim: PipelineClaim, custom_query: str = "") -> str:
        """
        Generate a query given a claim, the claim should not be preprocessed
        """
        # Construct a basic claim with cleaned text from the original claim
        query = self.__clean(claim.original_claim.claim + ' ' + claim.original_claim.claimant)
        # Add quoted strings
        query += f" {' '.join(self.__get_quoted_strs(claim.original_claim.claim))} "
        # Add preprocessed claim
        query += f" {claim.preprocessed_claim} "
        # Add items from a custom query string, if provided
        query += f" {custom_query}"
        return query

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

    def __get_quoted_strs(self, text: str) -> List[str]:
        """
        Get a list of quotes from the claim, we want to preserve these as is in case we match the quote directly
        """
        # Strip quotes
        return re.findall(r'\"(.+)\"', text)
