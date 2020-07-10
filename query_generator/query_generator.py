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
        # Query is made of 3 components - quoted strings, the original claim + claimant, and the preprocessed claim
        query = claim.original_claim.claim + ' ' + claim.original_claim.claimant
        query += ' ' + ' '.join(self.__get_quoted_strs(claim.original_claim.claim))
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

    def __get_quoted_strs(self, text: str) -> List[str]:
        """
        Get a list of quotes from the claim, we want to preserve these as is in case we match the quote directly
        """
        # Strip quotes
        return re.findall(r'\"(.+)\"', text)
