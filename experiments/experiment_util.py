import csv
import json
import os

import pandas as pd

from models import LeadersPrizeClaim
from preprocessor.html_preprocessor import HTMLProcessor
from preprocessor.text_preprocessor import TextPreprocessor
from query_generator.query_generator import QueryGenerator
from relevance_scorer.relevance_scorer import RelevanceScorer
from relevance_scorer.word2vec_vectorizer import Word2VecVectorizer
from search_client.client import ArticleSearchClient
from datetime import datetime


def get_query_generator():
    return QueryGenerator()


def get_search_client():
    return ArticleSearchClient('http://lpsa.wrw.org', 'ff5fdad7-de1f-4a74-bfac-acd42538131f')


def get_html_preprocessor():
    return HTMLProcessor()


def get_text_preprocessor():
    return TextPreprocessor()


def get_relevance_scorer():
    vectorizer = Word2VecVectorizer("../assets/GoogleNewsVectors.bin.gz")
    return RelevanceScorer(vectorizer)


def train_data_generator():
    with open("/Users/frankjia/Desktop/LeadersPrize/train/train.json") as json_file:
        data = json.load(json_file)
        for item in data:
            yield LeadersPrizeClaim(item)


def save_results(df: pd.DataFrame, tag: str, filekey: str):
    time_str = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    filename = f"{filekey}_{time_str}"
    filepath = f"output/{tag}/{filename}.csv"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, quoting=csv.QUOTE_ALL)
