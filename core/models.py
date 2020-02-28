from typing import List, Optional

from reasoner.transformers.models import Entailment, StsSimilarity
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
        self.id = data.get("id", "")
        self.claim = data.get("claim", "")
        self.claimant = data.get("claimant", "")
        self.date = data.get("date", "")
        # Optional fields
        self.label = data.get("label", None)
        related_articles = []
        data_related_articles = data.get("related_articles", None)
        if data_related_articles:
            for k, v in data_related_articles.items():
                related_articles.append(LeadersPrizeClaim.LeadersPrizeRelatedArticle(k, v))
        self.related_articles = related_articles


class PipelineClaim:
    """
    Base object for the pipeline. All services will annotate this object
    """

    def __init__(self, original_claim: LeadersPrizeClaim):
        # Original claim object
        self.original_claim = original_claim
        self.preprocessed_claim: str = ""
        self.articles: List[PipelineArticle] = []  # Results from search client
        self.articles_for_reasoner: List[PipelineArticle] = []  # Curated articles for the reasoner


class PipelineArticle:
    """
    An article for a given claim - includes all the preprocessed annotations
    """

    def __init__(self, raw_result: SearchQueryResult):
        self.id: str = ""
        self.raw_result = raw_result  # Raw HTML from client
        self.raw_body_text = None  # Raw body text parsed from raw result
        self.relevance = 0  # Relevance score of the article
        self.html_attributes = None  # Parsed HTML attributes
        self.preprocessed_sentences: List[PipelineSentence] = []  # Preprocessed sentences
        self.sentences_for_reasoner: List[PipelineSentence] = []  # Sentences extracted for processing by reasoner
        self.entailment_score: Optional[Entailment] = None  # Entailment as accessed by the reasoner


class PipelineSentence:
    """
    A sentence object, with annotations
    """

    def __init__(self, sentence: str):
        self.id: str = ""
        self.sentence: str = sentence
        self.relevance: float = 0  # Relevance score of the sentence
        self.sts_relevance_score: Optional[StsSimilarity] = None
        self.entailment_score: Optional[Entailment] = None  # Entailment as accessed by the reasoner

    def __repr__(self):
        return self.sentence + f"; Relevance: {self.relevance}; Entailment: {self.entailment_score}"
