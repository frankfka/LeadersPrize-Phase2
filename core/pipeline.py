from datetime import datetime
from typing import List

from analyze.document_relevance_scorer.lsa_document_relevance_scorer import LSADocumentRelevanceAnalyzer
from analyze.relevant_information_extractor.relevant_information_extractor import RelevantInformationExtractor
from core.models import LeadersPrizeClaim, PipelineClaim, PipelineArticle, PipelineSentence
from preprocess.html_preprocessor import HTMLProcessor
from preprocess.text_preprocessor import TextPreprocessor
from query_generator.query_generator import QueryGenerator
from analyze.sentence_relevance_scorer.word2vec_relevance_scorer import Word2VecRelevanceScorer
from analyze.sentence_relevance_scorer.word2vec_vectorizer import Word2VecVectorizer
from search_client.client import ArticleSearchClient

# TODO: These should be part of config
MIN_SENT_LEN = 5
NUM_ARTICLES_TO_PROCESS = 5
NUM_SENTS_PER_ARTICLE = 5
EXTRACT_LEFT_WINDOW = 1
EXTRACT_RIGHT_WINDOW = 2

class LeadersPrizePipeline:
    """
    Main pipeline for Leader's Prize
    """
    CONFIG_API_KEY = "api_key"
    CONFIG_ENDPOINT = "endpoint"
    CONFIG_W2V_PATH = "w2v_path"
    CONFIG_DEBUG = "debug"  # Prints debug info

    def __init__(self, config):
        self.debug_mode = config.get(LeadersPrizePipeline.CONFIG_DEBUG, False)
        self.search_client = ArticleSearchClient(config[LeadersPrizePipeline.CONFIG_ENDPOINT],
                                                 config[LeadersPrizePipeline.CONFIG_API_KEY])
        self.query_generator = QueryGenerator()
        self.article_relevance_scorer = LSADocumentRelevanceAnalyzer()
        self.html_preprocessor = HTMLProcessor()
        self.text_preprocessor = TextPreprocessor()
        w2v_vectorizer = Word2VecVectorizer(path=config[LeadersPrizePipeline.CONFIG_W2V_PATH])
        self.sentence_relevance_scorer = Word2VecRelevanceScorer(vectorizer=w2v_vectorizer)
        self.information_extractor = RelevantInformationExtractor()

    def predict(self, raw_claims: List[LeadersPrizeClaim]) -> List[PipelineClaim]:
        pipeline_objects: List[PipelineClaim] = []
        for claim in raw_claims:
            t = datetime.now()
            # Create pipeline object - this will hold all the annotations of our processing
            pipeline_object: PipelineClaim = PipelineClaim(claim)

            if self.debug_mode:
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

            # 2. Execute search query to get articles
            search_response = self.search_client.search(search_query)
            if search_response.error:
                # Error, the articles will just be empty
                print(f"Error searching query for claim {pipeline_object.original_claim.id}")
                # TODO: predict something and continue, or put on a retry count

            if self.debug_mode:
                nt = datetime.now()
                print(f"Retrieved articles for claim in {nt - t}")
                print(f"Query: {search_query}")
                print(f"{len(search_response.results)} Articles retrieved")
                print("\n")
                t = nt

            # 3. Process articles from raw HTML to parsed text
            pipeline_articles: List[PipelineArticle] = []
            for raw_article in search_response.results:
                pipeline_article = PipelineArticle(raw_article)
                # 3.1 Extract data from HTML
                html_process_result = self.html_preprocessor.process(pipeline_article.raw_result.content)
                pipeline_article.html_attributes = html_process_result.html_atts
                pipeline_article.raw_body_text = html_process_result.text
                pipeline_articles.append(pipeline_article)

            if self.debug_mode:
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

            if self.debug_mode:
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

            if self.debug_mode:
                nt = datetime.now()
                print(f"Preprocessed article text in {nt - t}")
                # print("Example preprocessed article")
                # print(pipeline_articles[0].preprocessed_sentences)
                print("\n")
                t = nt

            pipeline_object.articles = pipeline_articles
            # Use article relevance to only consider a subset of the most relevant articles from the ~30 given
            pipeline_articles.sort(key=lambda x: x.relevance, reverse=True)
            pipeline_object.articles_for_reasoner = pipeline_articles[0:NUM_ARTICLES_TO_PROCESS] if \
                len(pipeline_articles) > NUM_ARTICLES_TO_PROCESS else pipeline_articles

            # 6. Minor preprocessing for reasoner
            for article in pipeline_object.articles_for_reasoner:
                # 6.1 For each article, get most relevant sentence "blocks" for the reasoner to process
                extracted_sentences = self.information_extractor.extract(article.preprocessed_sentences,
                                                                         left_window=EXTRACT_LEFT_WINDOW,
                                                                         right_window=EXTRACT_RIGHT_WINDOW)
                if len(extracted_sentences) > NUM_SENTS_PER_ARTICLE:
                    extracted_sentences = extracted_sentences[0:NUM_SENTS_PER_ARTICLE]
                article.sentences_for_reasoner = extracted_sentences

            if self.debug_mode:
                nt = datetime.now()
                print(f"Reasoner preprocessing done in {nt - t}")
                print("Example sentence for reasoner")
                print(f"Article Relevance: {pipeline_object.articles_for_reasoner[0].relevance}")
                print(pipeline_object.articles_for_reasoner[0].sentences_for_reasoner[0])
                print("\n")
                t = nt

            pipeline_objects.append(pipeline_object)

        return pipeline_objects
