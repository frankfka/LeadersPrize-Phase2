import math
from datetime import datetime
from enum import Enum
from typing import List

from analyze.document_relevance_scorer.lsa_document_relevance_scorer import LSADocumentRelevanceAnalyzer
from analyze.relevant_information_extractor.relevant_information_extractor import RelevantInformationExtractor
from analyze.sentence_relevance_scorer.word2vec_relevance_scorer import Word2VecRelevanceScorer
from analyze.sentence_relevance_scorer.word2vec_vectorizer import Word2VecVectorizer
from analyze.supporting_evidence_generator.supporting_evidence_generator import SupportingEvidenceGenerator
from core.models import LeadersPrizeClaim, PipelineClaim, PipelineArticle, PipelineSentence
from preprocess import text_util
from preprocess.html_preprocessor import HTMLProcessor
from preprocess.text_preprocessor import TextPreprocessor
from query_generator.query_generator import QueryGenerator
from reasoner.transformer_reasoner import TransformerReasoner
from search_client.client import ArticleSearchClient


class PipelineConfigKeys(Enum):
    API_KEY = "api_key"
    ENDPOINT = "endpoint"
    # Whether to retrieve articles from search client, set to False to load from given training data
    RETRIEVE_ARTICLES = "retrieve_articles"
    W2V_PATH = "w2v_path"
    # For the transformer model
    TRANSFORMER_PATH = "transformer_path"
    DEBUG_MODE = "debug"  # Whether to print debug info

    NUM_SEARCH_ARTICLES = "num_search_articles"  # Number of articles to retrieve from search API
    MIN_SENT_LEN = "min_sent_len"  # Minimum length of sentence (in words) to consider
    SENT_RELEVANCE_CUTOFF = "sent_relevance_cutoff"  # Minimum w2v relevance
    NUM_ARTICLES_TO_PROCESS = "num_relevant_articles"  # Number of articles (with top relevance) to process
    NUM_SENTS_PER_ARTICLE = "num_sentences_per_article"  # Number of sentences per article to process
    EXTRACT_LEFT_WINDOW = "info_left_window"  # Information extraction - left window
    EXTRACT_RIGHT_WINDOW = "info_right_window"  # Information extraction - right window


class LeadersPrizePipeline:
    """
    Main pipeline for Leader's Prize
    """

    def __init__(self, config):
        self.config = config
        # Create inner dependencies
        self.search_client = ArticleSearchClient(config[PipelineConfigKeys.ENDPOINT],
                                                 config[PipelineConfigKeys.API_KEY])
        import spacy
        spacy_model = spacy.load("en_core_web_lg")
        self.query_generator = QueryGenerator(spacy_model)
        self.article_relevance_scorer = LSADocumentRelevanceAnalyzer()
        self.html_preprocessor = HTMLProcessor(debug=config[PipelineConfigKeys.DEBUG_MODE])
        self.text_preprocessor = TextPreprocessor()
        w2v_vectorizer = Word2VecVectorizer(path=config[PipelineConfigKeys.W2V_PATH])
        self.sentence_relevance_scorer = Word2VecRelevanceScorer(vectorizer=w2v_vectorizer)
        # from analyze.sentence_relevance_scorer.spacy_relevance_scorer import SpacyRelevanceScorer
        # self.sentence_relevance_scorer = SpacyRelevanceScorer(spacy_model)
        self.information_extractor = RelevantInformationExtractor(self.sentence_relevance_scorer)
        self.transformer_reasoner = TransformerReasoner(model_path=config[PipelineConfigKeys.TRANSFORMER_PATH],
                                                        debug=config[PipelineConfigKeys.DEBUG_MODE])
        self.support_generator = SupportingEvidenceGenerator()

    def predict(self, raw_claims: List[LeadersPrizeClaim]) -> List[PipelineClaim]:
        t = datetime.now()

        # Get some stuff from the config
        debug_mode = self.config.get(PipelineConfigKeys.DEBUG_MODE, False)
        num_articles_to_search = self.config.get(PipelineConfigKeys.NUM_SEARCH_ARTICLES, 30)
        min_sentence_length = self.config.get(PipelineConfigKeys.MIN_SENT_LEN, 5)
        num_relevant_articles_to_process = self.config.get(PipelineConfigKeys.NUM_ARTICLES_TO_PROCESS, 10)
        num_sentences_per_article_to_process = self.config.get(PipelineConfigKeys.NUM_SENTS_PER_ARTICLE, 10)
        min_sent_relevance = self.config.get(PipelineConfigKeys.SENT_RELEVANCE_CUTOFF, 0.5)
        info_extraction_left_window = self.config.get(PipelineConfigKeys.EXTRACT_LEFT_WINDOW, 1)
        info_extraction_right_window = self.config.get(PipelineConfigKeys.EXTRACT_RIGHT_WINDOW, 1)

        pipeline_objects_with_err: List[PipelineClaim] = []
        pipeline_objects_for_prediction: List[PipelineClaim] = []
        for (claim_idx, claim) in enumerate(raw_claims):

            if debug_mode:
                print(f"Processing claim {claim_idx}")

            # Create pipeline object - this will hold all the annotations of our processing
            pipeline_object: PipelineClaim = PipelineClaim(claim)
            pipeline_object.submission_id = pipeline_object.original_claim.id

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
            search_query = self.query_generator.get_query(pipeline_object)

            # 2. Execute search query to get articles if config allows
            if self.config.get(PipelineConfigKeys.RETRIEVE_ARTICLES, True):
                searched_articles = self.search_client.search(search_query, num_results=num_articles_to_search)
                if len(searched_articles) == 0:
                    # Error, predict defaults and move on
                    print(f"Error searching query for claim {pipeline_object.original_claim.id}")
                    pipeline_object.submission_explanation = "Error calling search API. No articles were retrieved."
                    pipeline_objects_with_err.append(pipeline_object)
                    continue
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
            article_relevances = self.article_relevance_scorer.analyze(claim=pipeline_object.preprocessed_claim,
                                                                       articles=[p.raw_body_text for p in pipeline_articles])
            for article_relevance, pipeline_article in zip(article_relevances, pipeline_articles):
                # Sometimes we get nan from numpy operations - default to 0 (irrelevant)
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

            # 5. Preprocess select articles on a sentence-level & annotate with sentence-level relevance
            all_sentences: List[PipelineSentence] = []
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
                article_sentences.sort(key=lambda sentence: sentence.relevance, reverse=True)
                # Only consider top n sentences per article
                if len(article_sentences) > num_sentences_per_article_to_process:
                    article_sentences = article_sentences[0:num_sentences_per_article_to_process]
                all_sentences += article_sentences

            # Extract sentences for transformer
            pipeline_object.sentences_for_transformer = self.information_extractor.extract_for_transformer(
                claim_str=pipeline_object.preprocessed_claim, supporting_sentences=all_sentences,
                min_sent_relevance=min_sent_relevance, deduplication_relevance_cutoff=0.97, max_num_words=500
            )

            # If no relevant information was extracted - write explanation and continue
            if len(pipeline_object.sentences_for_transformer) == 0:
                pipeline_object.submission_explanation = "No relevant information was found to support the claim."
                pipeline_objects_with_err.append(pipeline_object)
                continue

            if debug_mode:
                nt = datetime.now()
                print(f"Preprocessed article text in {nt - t}")
                # print("Example preprocessed article")
                # print(pipeline_articles[0].preprocessed_sentences)
                print("\n")
                t = nt

            # Append for batch predictions
            pipeline_objects_for_prediction.append(pipeline_object)

        # 6. Batch predictions from transformers
        predictions = self.transformer_reasoner.predict(pipeline_objects_for_prediction)
        for (pipeline_object, prediction) in zip(pipeline_objects_for_prediction, predictions):
            # Populate submission values
            explanation, supporting_urls = self.support_generator.get_evidence(pipeline_object)
            pipeline_object.submission_label = prediction.value
            pipeline_object.submission_article_urls = supporting_urls
            pipeline_object.submission_explanation = explanation

            if debug_mode:
                nt = datetime.now()
                print(f"Reasoner predicted in {nt - t}")
                print(f"Prediction: {pipeline_object.submission_label}")
                print("\n")
                t = nt

        return pipeline_objects_for_prediction + pipeline_objects_with_err
