from models import LeadersPrizeClaim
import nltk

from util.text_util import clean_tokenized


class QueryGenerator:
    """
    Generates a query based on all words in claim + claimant, stripping punctuation and stopwords
    """

    def __init__(self):
        nltk.download('stopwords')

    def get_query(self, claim: LeadersPrizeClaim) -> str:
        return self.__clean(claim.claim + ' ' + claim.claimant)

    def __clean(self, text: str) -> str:
        """
        Cleans text given by the claim
        """
        tokens = nltk.tokenize.word_tokenize(text)
        tokens = clean_tokenized(tokens, lowercase=True, remove_punctuation=True, remove_stopwords=True)
        # Remove numbers and short words - numbers match HTML elements
        tokens_cleaned = [tok for tok in tokens if not tok.isnumeric() and len(tok) > 2]
        # TODO: Can look at removing duplicate words
        if tokens_cleaned:
            return ' '.join(tokens_cleaned)
        return ' '.join(tokens)
