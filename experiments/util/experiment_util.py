import csv
import os

import pandas as pd

from analyze.relevant_information_extractor.relevant_information_extractor import RelevantInformationExtractor
from analyze.sentence_relevance_scorer.infersent_relevance_scorer import InfersentRelevanceScorer
from analyze.sentence_relevance_scorer.infersent_vectorizer import InfersentVectorizer
from analyze.truth_tuple_extractor.truth_tuple_extractor import TruthTupleExtractor
from preprocess.html_preprocessor import HTMLProcessor
from preprocess.text_preprocessor import TextPreprocessor
from query_generator.query_generator import QueryGenerator
from analyze.sentence_relevance_scorer.word2vec_relevance_scorer import Word2VecRelevanceScorer
from analyze.sentence_relevance_scorer.word2vec_vectorizer import Word2VecVectorizer
from search_client.client import ArticleSearchClient
from datetime import datetime


def get_query_generator():
    return QueryGenerator()


def get_search_client():
    return ArticleSearchClient('http://lpsa.wrw.org', 'ff5fdad7-de1f-4a74-bfac-acd42538131f')


def get_html_preprocessor():
    return HTMLProcessor()


def get_truth_tuple_extractor():
    return TruthTupleExtractor()


def get_text_preprocessor():
    return TextPreprocessor()


def get_doc_sentence_extractor():
    return RelevantInformationExtractor()


def get_infersent_relevance_scorer():
    vectorizer = InfersentVectorizer("/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/assets/infersent"
                                     "/infersent2.pkl",
                                     "/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/assets/fastttext/crawl"
                                     "-300d-2M.vec")
    return InfersentRelevanceScorer(vectorizer)


def get_word2vec_relevance_scorer():
    vectorizer = Word2VecVectorizer("/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/assets/word2vec"
                                    "/GoogleNewsVectors.bin.gz")
    return Word2VecRelevanceScorer(vectorizer)


def save_results(df: pd.DataFrame, tag: str, filekey: str, time_str: str = None):
    if not time_str:
        time_str = get_timestamp()
    filename = f"{filekey}_{time_str}"
    filepath = f"output/{tag}/{filename}.csv"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, quoting=csv.QUOTE_ALL)


def get_timestamp() -> str:
    return datetime.now().strftime("%m-%d-%Y_%H-%M-%S")