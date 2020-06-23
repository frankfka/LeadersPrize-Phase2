from datetime import datetime
from enum import Enum
from random import randint
from typing import List
import math

from analyze.document_relevance_scorer.lsa_document_relevance_scorer import LSADocumentRelevanceAnalyzer
from analyze.relevant_information_extractor.relevant_information_extractor import RelevantInformationExtractor
from analyze.sentence_relevance_scorer.word2vec_relevance_scorer import Word2VecRelevanceScorer
from analyze.sentence_relevance_scorer.word2vec_vectorizer import Word2VecVectorizer
from core.models import LeadersPrizeClaim, PipelineClaim, PipelineArticle, PipelineSentence
from preprocess.html_preprocessor import HTMLProcessor
from preprocess.text_preprocessor import TextPreprocessor
from query_generator.query_generator import QueryGenerator
from reasoner.preprocess import get_text_b_for_reasoner
from reasoner.transformer_reasoner import TransformerReasoner
from search_client.client import ArticleSearchClient


# TODO: Increase # articles to process
# TODO: Get stuff for reasoner by blending articles, getting highest relevance?

class PipelineConfigKeys(Enum):
    API_KEY = "api_key"
    ENDPOINT = "endpoint"
    # Whether to retrieve articles from search client, set to False to load from given training data
    RETRIEVE_ARTICLES = "retrieve_articles"
    W2V_PATH = "w2v_path"
    # For the transformer model
    TRANSFORMER_PATH = "transformer_path"
    DEBUG_MODE = "debug"  # Whether to print debug info

    MIN_SENT_LEN = "min_sent_len"  # Minimum length of sentence (in words) to consider
    NUM_ARTICLES_TO_PROCESS = "num_relevant_articles"  # Number of articles (with top relevance) to process
    NUM_SENTS_PER_ARTICLE = "num_sentences_per_article"  # Number of sentences per article to process
    EXTRACT_LEFT_WINDOW = "info_left_window"  # Information extraction - left window
    EXTRACT_RIGHT_WINDOW = "info_right_window"  # Information extraction - right wondpw


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
        self.transformer_reasoner = TransformerReasoner(model_path=config[PipelineConfigKeys.TRANSFORMER_PATH],
                                                        debug=config[PipelineConfigKeys.DEBUG_MODE])

    def predict(self, raw_claims: List[LeadersPrizeClaim]) -> List[PipelineClaim]:
        t = datetime.now()

        # Get some stuff from the config
        debug_mode = self.config.get(PipelineConfigKeys.DEBUG_MODE, False)
        min_sentence_length = self.config.get(PipelineConfigKeys.MIN_SENT_LEN, 5)
        num_relevant_articles_to_process = self.config.get(PipelineConfigKeys.NUM_ARTICLES_TO_PROCESS, 10)
        num_sentences_per_article_to_process = self.config.get(PipelineConfigKeys.NUM_SENTS_PER_ARTICLE, 5)
        info_extraction_left_window = self.config.get(PipelineConfigKeys.EXTRACT_LEFT_WINDOW, 1)
        info_extraction_right_window = self.config.get(PipelineConfigKeys.EXTRACT_RIGHT_WINDOW, 1)

        pipeline_objects: List[PipelineClaim] = []
        for claim in raw_claims:
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
            claim_with_claimant = claim.claimant + " " + claim.claim
            processed_claim = self.text_preprocessor.process(claim_with_claimant)
            if len(processed_claim.sentences) > 0:
                pipeline_object.preprocessed_claim = processed_claim.sentences[0]
            else:
                print("Preprocessed claim is empty - defaulting to original claim")
                pipeline_object.preprocessed_claim = claim_with_claimant

            # 2. Execute search query to get articles if config allows
            if self.config.get(PipelineConfigKeys.RETRIEVE_ARTICLES, True):
                searched_articles = self.search_client.search(search_query)
                if len(searched_articles) == 0:
                    # Error, the articles will just be empty
                    print(f"Error searching query for claim {pipeline_object.original_claim.id}")
                    # TODO: make sure we still predict something
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
                # Sometimes we get nan from numpy operations
                pipeline_article.relevance = article_relevance if math.isfinite(article_relevance) else 0

            if debug_mode:
                nt = datetime.now()
                print(f"Analyzed article relevances in {nt - t}")
                print(f"Maximum article relevance found: {max(map(lambda x: x.relevance, pipeline_articles))}")
                print("\n")
                t = nt

            # 4.5 Based on article relevance, only consider the top relevances
            pipeline_articles.sort(key=lambda article: article.relevance, reverse=True)
            if len(pipeline_articles) > num_relevant_articles_to_process:
                pipeline_articles = pipeline_articles[:num_relevant_articles_to_process - 1]

            # 5. Preprocess select articles on a sentence-level & annotate with sentence-level relevance
            for pipeline_article in pipeline_articles:
                # 5.1 Clean text data
                text_process_result = self.text_preprocessor.process(pipeline_article.raw_body_text)
                # 5.2 Get relevances for each sentence
                article_sentences: List[PipelineSentence] = []
                for preprocessed_sentence in text_process_result.sentences:
                    # Enforce a minimum sentence length
                    if len(preprocessed_sentence.split()) < min_sentence_length:
                        continue
                    relevance = self.sentence_relevance_scorer.get_relevance(pipeline_object.preprocessed_claim,
                                                                             preprocessed_sentence)
                    pipeline_sentence = PipelineSentence(preprocessed_sentence)
                    pipeline_sentence.relevance = relevance
                    # Attach the parent URL to the sentence so we can trace it back
                    pipeline_sentence.parent_article_url = pipeline_article.url
                    article_sentences.append(pipeline_sentence)
                pipeline_article.preprocessed_sentences = article_sentences
                # 5.3 Get select sentences for reasoner, then cut to the most relevant sentences
                sentences_for_reasoner = self.information_extractor.extract(article_sentences,
                                                                            left_window=info_extraction_left_window,
                                                                            right_window=info_extraction_right_window)
                sentences_for_reasoner.sort(key=lambda sentence: sentence.relevance, reverse=True)
                if len(sentences_for_reasoner) > num_sentences_per_article_to_process:
                    sentences_for_reasoner = sentences_for_reasoner[0:num_sentences_per_article_to_process - 1]
                pipeline_article.sentences_for_reasoner = sentences_for_reasoner
            pipeline_object.articles_for_reasoner = pipeline_articles

            # 5.4 Get cumulative text_b for reasoner
            text_b, reasoner_article_urls = get_text_b_for_reasoner(pipeline_object)
            pipeline_object.preprocessed_text_b_for_reasoner = text_b

            if debug_mode:
                nt = datetime.now()
                print(f"Preprocessed article text in {nt - t}")
                # print("Example preprocessed article")
                # print(pipeline_articles[0].preprocessed_sentences)
                print("\n")
                t = nt

            # Populate submission values
            pipeline_object.submission_id = pipeline_object.original_claim.id
            if len(reasoner_article_urls) > 2:
                reasoner_article_urls = reasoner_article_urls[0:2]
            pipeline_object.submission_article_urls = reasoner_article_urls

            # Construct the explanation
            def get_explanation(source_str: str) -> str:
                source_sents = source_str.split("$.$")
                explanation = "The articles state that: "
                for sent in source_sents:
                    if len(explanation) > 1000:
                        break
                    explanation += f" {sent} . "
                if len(explanation) >= 1000:
                    explanation = explanation[0:999]
                return explanation
            pipeline_object.submission_explanation = get_explanation(text_b)

            # Delete unneeded stuff to save memory
            del pipeline_object.articles
            del pipeline_object.articles_for_reasoner

            pipeline_objects.append(pipeline_object)

        # 6. Batch predictions from transformers
        predictions = self.transformer_reasoner.predict(pipeline_objects)
        for (pipeline_object, prediction) in zip(pipeline_objects, predictions):
            # Populate label
            pipeline_object.submission_label = prediction.value

            if debug_mode:
                nt = datetime.now()
                print(f"Reasoner predicted in {nt - t}")
                print(f"Prediction: {pipeline_object.submission_label}")
                print("\n")
                t = nt

        return pipeline_objects
