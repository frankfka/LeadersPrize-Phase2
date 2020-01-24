from models import LeadersPrizeClaim
import nltk

import util.text_util as text_util


class QueryGenerator:
    """
    Generates a query based on all words in claim + claimant, stripping punctuation and stopwords
    """

    def get_query(self, claim: LeadersPrizeClaim) -> str:
        return self.__clean(claim.claim + ' ' + claim.claimant)

    def __clean(self, text: str) -> str:
        """
        Cleans text given by the claim
        """
        tokens = text_util.tokenize_by_word(text)
        tokens = text_util.clean_tokenized(tokens, lowercase=True, remove_punctuation=True, remove_stopwords=True)
        # Remove numbers and short words - numbers match HTML elements
        tokens_cleaned = [tok for tok in tokens if not tok.isnumeric() and len(tok) > 2]
        if tokens_cleaned:
            return ' '.join(tokens_cleaned)
        return ' '.join(tokens)
