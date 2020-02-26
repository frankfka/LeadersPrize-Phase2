from datetime import datetime
from typing import List

from analyze.document_relevance_scorer.lsa_document_relevance_scorer import LSADocumentRelevanceAnalyzer
from analyze.relevant_information_extractor.relevant_information_extractor import RelevantInformationExtractor
from analyze.truth_tuple_extractor.truth_tuple_extractor import TruthTupleExtractor
from core.models import LeadersPrizeClaim, PipelineClaim, PipelineArticle, PipelineSentence
from preprocess.html_preprocessor import HTMLProcessor
from preprocess.text_preprocessor import TextPreprocessor
from query_generator.query_generator import QueryGenerator
from analyze.sentence_relevance_scorer.word2vec_relevance_scorer import Word2VecRelevanceScorer
from analyze.sentence_relevance_scorer.word2vec_vectorizer import Word2VecVectorizer
from search_client.client import ArticleSearchClient


class LeadersPrizePipeline:
    """
    Main pipeline for Leader's Prize
    - TODO: Deal with cases where we have 0 articles, currently throws exception
    """
    CONFIG_API_KEY = "api_key"
    CONFIG_ENDPOINT = "endpoint"
    CONFIG_W2V_PATH = "w2v_path"
    CONFIG_DEBUG = "debug"  # Prints debug info

    def __init__(self, config):
        self.debug_mode = config.get(LeadersPrizePipeline.CONFIG_DEBUG, False)
        self.search_client = ArticleSearchClient(config[LeadersPrizePipeline.CONFIG_ENDPOINT],
                                                 config[LeadersPrizePipeline.CONFIG_API_KEY])
        self.truth_tuple_extractor = TruthTupleExtractor()
        self.query_generator = QueryGenerator()
        self.article_relevance_scorer = LSADocumentRelevanceAnalyzer()
        self.html_preprocessor = HTMLProcessor()
        self.text_preprocessor = TextPreprocessor()
        w2v_vectorizer = Word2VecVectorizer(path=config[LeadersPrizePipeline.CONFIG_W2V_PATH])
        self.sentence_relevance_scorer = Word2VecRelevanceScorer(vectorizer=w2v_vectorizer)
        self.bert_information_extractor = RelevantInformationExtractor()

    def predict(self, raw_claims: List[LeadersPrizeClaim]) -> List[PipelineClaim]:
        pipeline_objects: List[PipelineClaim] = []
        for claim in raw_claims:
            t = datetime.now()
            # Create pipeline object - this will hold all the annotations of our processing
            pipeline_object: PipelineClaim = PipelineClaim(claim)
            claim_with_claimant = pipeline_object.original_claim.claimant + " " + pipeline_object.original_claim.claim
            claim_truth_tuples = self.truth_tuple_extractor.extract(claim_with_claimant)
            pipeline_object.claim_truth_tuples = claim_truth_tuples

            if self.debug_mode:
                nt = datetime.now()
                print(f"Initialized claim in {nt - t}")
                print(f"Claim with Claimant: {claim_with_claimant}")
                print(f"Parsed Truth Tuples: {pipeline_object.claim_truth_tuples}")
                print("\n")
                t = nt

            # 1. Get query from claim
            # - Note: not using truth tuples for now, given that we see no significant difference with them
            search_query = self.query_generator.get_query(pipeline_object.original_claim,
                                                          truth_tuples=[])
            # 1.1 Preprocess the claim + claimant
            processed_claim = self.text_preprocessor.process(claim_with_claimant)
            if len(processed_claim.bert_sentences) > 0:
                pipeline_object.preprocessed_claim = processed_claim.bert_sentences[0]
            else:
                print("Preprocessed claim is empty - defaulting to original claim")
                pipeline_object.preprocessed_claim = claim_with_claimant

            # 2. Execute search query to get articles
            search_response = self.search_client.search(search_query)
            if search_response.error:
                # Error, the articles will just be empty
                print(f"Error searching query for claim {pipeline_object.original_claim.id}")

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
                print("== First parsed article ==")
                print(pipeline_articles[0].raw_body_text)
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

            # 5. Preprocess select articles for BERT & annotate with sentence-level relevance
            for pipeline_article in pipeline_articles:
                # 5.1 Clean text data
                text_process_result = self.text_preprocessor.process(pipeline_article.raw_body_text)
                # 5.2 Get relevances for each sentence
                article_sentences: List[PipelineSentence] = []
                for bert_sentence in text_process_result.bert_sentences:
                    # Enforce a minimum sentence length
                    if len(bert_sentence.split()) < 5:
                        continue
                    relevance = self.sentence_relevance_scorer.get_relevance(pipeline_object.preprocessed_claim,
                                                                             bert_sentence)
                    pipeline_sentence = PipelineSentence(bert_sentence)
                    pipeline_sentence.relevance = relevance
                    article_sentences.append(pipeline_sentence)
                pipeline_article.preprocessed_sentences = article_sentences

            if self.debug_mode:
                nt = datetime.now()
                print(f"Preprocessed article text in {nt - t}")
                print("Example preprocessed article")
                print(pipeline_articles[0].preprocessed_sentences)
                print("\n")
                t = nt

            # Assign the preprocessed & annotated result to the pipeline object
            pipeline_object.articles = pipeline_articles

            # 6. Extract Information for BERT - concat all the sentences
            # Use article relevance to only consider the 5 most relevant articles from the ~30 given
            pipeline_articles.sort(key=lambda x: x.relevance, reverse=True)
            bert_articles = pipeline_articles[0:5] if len(pipeline_articles) > 5 else pipeline_articles
            bert_article_sentences = []
            for article in bert_articles:
                bert_article_sentences += article.preprocessed_sentences
            # Use large window to extract chunks of information around highly relevant sentences
            bert_sentences_by_relevance = self.bert_information_extractor.extract(bert_article_sentences, window=3)
            bert_preprocessed = ""
            num_words_in_bert_preprocessed = 0
            for bert_sentence in bert_sentences_by_relevance:
                if num_words_in_bert_preprocessed > 512:
                    break
                bert_preprocessed += ' . ' + bert_sentence.sentence
                num_words_in_bert_preprocessed += len(bert_sentence.sentence.split())
            pipeline_object.bert_preprocessed = bert_preprocessed

            nt = datetime.now()
            if self.debug_mode:
                print(f"Extracted information for BERT in {nt - t}")
            t = nt

            pipeline_objects.append(pipeline_object)

        # 7. Run predictive algorithms on pipeline objects
        # TODO

        return pipeline_objects
