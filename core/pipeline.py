from datetime import datetime
from enum import Enum
from typing import List
import math

from analyze.document_relevance_scorer.lsa_document_relevance_scorer import LSADocumentRelevanceAnalyzer
from analyze.ne_reasoner.predict_ensemble import EnsembleClassifier
from analyze.relevant_information_extractor.relevant_information_extractor import RelevantInformationExtractor
from analyze.sentence_relevance_scorer.word2vec_relevance_scorer import Word2VecRelevanceScorer
from analyze.sentence_relevance_scorer.word2vec_vectorizer import Word2VecVectorizer
from core.models import LeadersPrizeClaim, PipelineClaim, PipelineArticle, PipelineSentence
from preprocess import text_util
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
    # For NE's reasoner
    NE_REASONER_PATH = "reasoner_path"

    NUM_SEARCH_ARTICLES = "num_search_articles"  # Number of articles to retrieve from search API
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
        ne_reasoner_root_path = config[PipelineConfigKeys.NE_REASONER_PATH]
        self.ne_reasoner = EnsembleClassifier(
        ckpt_path="/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/assets/ne-reasoner/ckpt",
        cvm_path="/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/assets/ne-reasoner/bows_premise_space_cvm_utf8.csv",
        vocab_path="/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/assets/ne-reasoner/ensemble_vocab.p",
        glove_emb_path="/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/assets/ne-reasoner/glove.6B.50d.txt",
    )

    def predict(self, raw_claims: List[LeadersPrizeClaim]) -> List[PipelineClaim]:
        t = datetime.now()

        # Get some stuff from the config
        debug_mode = self.config.get(PipelineConfigKeys.DEBUG_MODE, False)
        num_articles_to_search = self.config.get(PipelineConfigKeys.NUM_SEARCH_ARTICLES, 60)
        min_sentence_length = self.config.get(PipelineConfigKeys.MIN_SENT_LEN, 5)
        num_relevant_articles_to_process = self.config.get(PipelineConfigKeys.NUM_ARTICLES_TO_PROCESS, 15)  # 23 min for 30 articles to process for 50 claims
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

            # 1. Preprocess the claim
            claim_with_claimant = claim.claimant + " " + claim.claim
            pipeline_object.preprocessed_claim = self.text_preprocessor.process_one_sentence(claim_with_claimant)

            # 1.1 Get query from claim
            # - Note: not using truth tuples for now, given that we see no significant difference with them
            search_query = self.query_generator.get_query(pipeline_object.original_claim, custom_query=pipeline_object.preprocessed_claim)

            # 2. Execute search query to get articles if config allows
            if self.config.get(PipelineConfigKeys.RETRIEVE_ARTICLES, True):
                searched_articles = self.search_client.search(search_query, num_results=num_articles_to_search)
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
            for searched_article in searched_articles:
                if searched_article and searched_article.content:
                    pipeline_article = PipelineArticle()
                    pipeline_article.url = searched_article.url
                    # 3.1 Extract data from HTML
                    pipeline_article.raw_body_text = self.html_preprocessor.process(searched_article.content).text
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
                if pipeline_articles:
                    print(f"Maximum article relevance found: {max(map(lambda x: x.relevance, pipeline_articles))}")
                print("\n")
                t = nt

            # 4.5 Based on article relevance, only consider the top relevances
            pipeline_articles.sort(key=lambda article: article.relevance, reverse=True)
            if len(pipeline_articles) > num_relevant_articles_to_process:
                pipeline_articles = pipeline_articles[:num_relevant_articles_to_process]

            article_preds_false = []
            article_preds_neu = []
            article_preds_true = []

            # 5. Preprocess select articles on a sentence-level & annotate with sentence-level relevance
            for pipeline_article in pipeline_articles:
                # 5.1 Clean text data
                original_sentences = text_util.tokenize_by_sentence(pipeline_article.raw_body_text)
                preprocessed_sentences = self.text_preprocessor.process_sentences(original_sentences)
                # 5.2 Get relevances for each sentence
                article_sentences: List[PipelineSentence] = []
                for (original_sentence, preprocessed_sentence) in zip(original_sentences, preprocessed_sentences):
                    # Enforce a minimum sentence length
                    if len(preprocessed_sentence.split()) < min_sentence_length:
                        continue
                    relevance = self.sentence_relevance_scorer.get_relevance(pipeline_object.preprocessed_claim,
                                                                             preprocessed_sentence)
                    pipeline_sentence = PipelineSentence()
                    pipeline_sentence.text = original_sentence
                    pipeline_sentence.preprocessed_text = preprocessed_sentence
                    pipeline_sentence.relevance = relevance
                    # Attach the parent URL to the sentence so we can trace it back
                    pipeline_sentence.parent_article_url = pipeline_article.url
                    article_sentences.append(pipeline_sentence)
                pipeline_article.sentences = article_sentences

                # Predict using reasoner
                article_str_sents = list(map(lambda x: x.text, article_sentences))
                pred_magnitudes, _, _ = self.ne_reasoner.predict_for_claim(pipeline_object.preprocessed_claim, article_str_sents)
                article_preds_false.append(pred_magnitudes[0])
                article_preds_neu.append(pred_magnitudes[1])
                article_preds_true.append(pred_magnitudes[2])

            if debug_mode:
                nt = datetime.now()
                print(f"Preprocessed article text in {nt - t}")
                # print("Example preprocessed article")
                # print(pipeline_articles[0].preprocessed_sentences)
                print("\n")
                t = nt

            # Populate submission values
            pipeline_object.submission_id = pipeline_object.original_claim.id
            pipeline_object.submission_article_urls = []

            # Get the final prediction
            max_false = max(article_preds_false)
            max_neu = max(article_preds_neu)
            max_true = max(article_preds_true)

            import numpy as np
            pipeline_object.submission_label = np.argmax([max_false, max_neu, max_true])

            pipeline_objects.append(pipeline_object)

        return pipeline_objects
