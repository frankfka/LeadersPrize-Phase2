from typing import List, Optional

from search_client.client import SearchQueryResult


class LeadersPrizeClaim:
    """
    Data given by the JSON file
    """

    class LeadersPrizeRelatedArticle:
        def __init__(self, dict_key, dict_val):
            self.filepath = dict_key
            self.url = dict_val

    def __init__(self, data):
        """
        Create claim from JSON data in metadata file
        """
        self.id: str = data.get("id", "")
        self.claim: str = data.get("claim", "")
        self.claimant: str = data.get("claimant", "")
        self.date: str = data.get("date", "")
        # Optional fields
        self.label: int = data.get("label", None)
        related_articles = []
        data_related_articles = data.get("related_articles", None)
        if data_related_articles:
            for k, v in data_related_articles.items():
                related_articles.append(LeadersPrizeClaim.LeadersPrizeRelatedArticle(k, v))
        self.related_articles: List[LeadersPrizeClaim.LeadersPrizeRelatedArticle] = related_articles
        # Optional if we want to run the pipeline without search client
        self.mock_search_results: List[SearchQueryResult] = []


class PipelineClaim:
    """
    Base object for the pipeline. All services will annotate this object
    """

    def __init__(self, original_claim: LeadersPrizeClaim):
        # Original claim object
        self.original_claim = original_claim
        self.preprocessed_claim: str = ""
        # Intermediate properties
        self.sentences_for_transformer: List[PipelineSentence] = []  # Curated sentences for the transformer
        # Submission properties
        self.submission_id: str = ""
        self.submission_label: int = 1
        self.submission_explanation: str = ""
        self.submission_article_urls: List[str] = []


class PipelineArticle:
    """
    An article for a given claim - includes all the preprocessed annotations
    """

    def __init__(self):
        self.id: str = ""
        self.url = ""
        self.raw_body_text = ""  # Raw body text parsed from search result
        self.relevance = 0  # Relevance score of the article
        self.sentences: List[PipelineSentence] = []  # All sentences


class PipelineSentence:
    """
    A sentence object, with annotations
    """

    def __init__(self):
        self.id: str = ""
        self.text: str = ""
        self.preprocessed_text: str = ""
        self.relevance: float = 0  # Relevance score of the sentence
        self.parent_article_url: Optional[str] = None

    def __repr__(self):
        return f"(Relevance: {self.relevance}) " + self.preprocessed_text
