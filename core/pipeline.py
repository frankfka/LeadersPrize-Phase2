from datetime import datetime
from enum import Enum
from random import randint
from typing import List

from analyze.document_relevance_scorer.lsa_document_relevance_scorer import LSADocumentRelevanceAnalyzer
from analyze.relevant_information_extractor.relevant_information_extractor import RelevantInformationExtractor
from analyze.sentence_relevance_scorer.word2vec_relevance_scorer import Word2VecRelevanceScorer
from analyze.sentence_relevance_scorer.word2vec_vectorizer import Word2VecVectorizer
from core.models import LeadersPrizeClaim, PipelineClaim, PipelineArticle, PipelineSentence
from preprocess.html_preprocessor import HTMLProcessor
from preprocess.text_preprocessor import TextPreprocessor
from query_generator.query_generator import QueryGenerator
from search_client.client import ArticleSearchClient

# TODO: These should be part of config
MIN_SENT_LEN = 5
NUM_ARTICLES_TO_PROCESS = 5
NUM_SENTS_PER_ARTICLE = 5
EXTRACT_LEFT_WINDOW = 0
EXTRACT_RIGHT_WINDOW = 1


class PipelineConfigKeys(Enum):
    API_KEY = "api_key"
    ENDPOINT = "endpoint"
    # Whether to retrieve articles from search client, set to False to load from given training data
    RETRIEVE_ARTICLES = "retrieve_articles"
    W2V_PATH = "w2v_path"
    DEBUG_MODE = "debug"  # Whether to print debug info


class LeadersPrizePipeline:
    """
    Main pipeline for Leader's Prize
    """

    def __init__(self, config):
        self.config = config
        # Create inner dependencies
        self.search_client = ArticleSearchClient(config[PipelineConfigKeys.ENDPOINT],
                                                 config[PipelineConfigKeys.API_KEY])
        self.query_generator = QueryGenerator()
        self.article_relevance_scorer = LSADocumentRelevanceAnalyzer()
        self.html_preprocessor = HTMLProcessor()
        self.text_preprocessor = TextPreprocessor()
        w2v_vectorizer = Word2VecVectorizer(path=config[PipelineConfigKeys.W2V_PATH])
        self.sentence_relevance_scorer = Word2VecRelevanceScorer(vectorizer=w2v_vectorizer)
        self.information_extractor = RelevantInformationExtractor()

    def predict(self, raw_claims: List[LeadersPrizeClaim]) -> List[PipelineClaim]:
        debug_mode = self.config.get(PipelineConfigKeys.DEBUG_MODE, False)
        pipeline_objects: List[PipelineClaim] = []
        for claim in raw_claims:
            t = datetime.now()
            # Create pipeline object - this will hold all the annotations of our processing
            pipeline_object: PipelineClaim = PipelineClaim(claim)

            if debug_mode:
                nt = datetime.now()
                print(f"Initialized claim in {nt - t}")
                print(f"Claimant: {claim.claimant}, Claim: {claim.claim}")
                print("\n")
                t = nt

            # 1. Get query from claim
            # - Note: not using truth tuples for now, given that we see no significant difference with them
            search_query = self.query_generator.get_query(pipeline_object.original_claim)
            # 1.1 Preprocess the claim
            # - Note: not appending the claimant as that may impact entailment
            processed_claim = self.text_preprocessor.process(claim.claim)
            if len(processed_claim.sentences) > 0:
                pipeline_object.preprocessed_claim = processed_claim.sentences[0]
            else:
                print("Preprocessed claim is empty - defaulting to original claim")
                pipeline_object.preprocessed_claim = claim.claim

            # 2. Execute search query to get articles if config allows
            if self.config.get(PipelineConfigKeys.RETRIEVE_ARTICLES, True):
                search_response = self.search_client.search(search_query)
                searched_articles = search_response.results
                if search_response.error or len(searched_articles) == 0:
                    # Error, the articles will just be empty
                    print(f"Error searching query for claim {pipeline_object.original_claim.id}")
                    # TODO: predict something and continue, or put on a retry count
            # 2. OR: if we're loading local articles
            else:
                searched_articles = claim.mock_search_results

            if debug_mode:
                nt = datetime.now()
                print(f"Retrieved articles for claim in {nt - t}")
                print(f"Query: {search_query}")
                print(f"{len(searched_articles)} Articles retrieved")
                print("\n")
                t = nt

            # 3. Process articles from raw HTML to parsed text
            pipeline_articles: List[PipelineArticle] = []
            for raw_article in searched_articles:
                if raw_article and raw_article.content:
                    pipeline_article = PipelineArticle(raw_article)
                    # 3.1 Extract data from HTML
                    html_process_result = self.html_preprocessor.process(raw_article.content)
                    pipeline_article.html_attributes = html_process_result.html_atts
                    pipeline_article.raw_body_text = html_process_result.text
                    pipeline_articles.append(pipeline_article)

            if debug_mode:
                nt = datetime.now()
                print(f"Analyzed article HTML in {nt - t}")
                # print("== First parsed article ==")
                # print(pipeline_articles[0].raw_body_text)
                print("\n")
                t = nt

            # 4. Get Article Relevance
            pipeline_article_texts: List[str] = [p.raw_body_text for p in pipeline_articles]
            article_relevances = self.article_relevance_scorer.analyze(pipeline_object.preprocessed_claim,
                                                                       pipeline_article_texts)
            for article_relevance, pipeline_article in zip(article_relevances, pipeline_articles):
                pipeline_article.relevance = article_relevance

            if debug_mode:
                nt = datetime.now()
                print(f"Analyzed article relevances in {nt - t}")
                print(f"Maximum article relevance found: {max(map(lambda x: x.relevance, pipeline_articles))}")
                print("\n")
                t = nt

            # 5. Preprocess select articles on a sentence-level & annotate with sentence-level relevance
            for pipeline_article in pipeline_articles:
                # 5.1 Clean text data
                text_process_result = self.text_preprocessor.process(pipeline_article.raw_body_text)
                # 5.2 Get relevances for each sentence
                article_sentences: List[PipelineSentence] = []
                for preprocessed_sentence in text_process_result.sentences:
                    # Enforce a minimum sentence length
                    if len(preprocessed_sentence.split()) < MIN_SENT_LEN:
                        continue
                    relevance = self.sentence_relevance_scorer.get_relevance(pipeline_object.preprocessed_claim,
                                                                             preprocessed_sentence)
                    pipeline_sentence = PipelineSentence(preprocessed_sentence)
                    pipeline_sentence.relevance = relevance
                    article_sentences.append(pipeline_sentence)
                pipeline_article.preprocessed_sentences = article_sentences

            if debug_mode:
                nt = datetime.now()
                print(f"Preprocessed article text in {nt - t}")
                # print("Example preprocessed article")
                # print(pipeline_articles[0].preprocessed_sentences)
                print("\n")
                t = nt

            # pipeline_object = self.reasoner.predict(pipeline_object)
            #
            # if debug_mode:
            #     nt = datetime.now()
            #     print(f"Reasoner predicted in {nt - t}")
            #     print(f"Prediction: {pipeline_object.submission_label}")
            #     print("\n")
            #     t = nt

            # TEMPORARY, get article urls
            reasoner_article_urls = [article.url for article in searched_articles]
            if len(reasoner_article_urls) > 2:
                reasoner_article_urls = reasoner_article_urls[0:2]

            # TEMPORARY: Test submission
            pipeline_object.submission_id = pipeline_object.original_claim.id
            pipeline_object.submission_article_urls = reasoner_article_urls
            pipeline_object.submission_label = randint(0,2)
            pipeline_object.submission_explanation = "Some explanation"

            pipeline_objects.append(pipeline_object)

        return pipeline_objects
