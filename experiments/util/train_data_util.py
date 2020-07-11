import json
import os
from typing import Optional

import pandas as pd

from core.models import LeadersPrizeClaim


def get_preprocessed_train_data(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)


def get_train_claims_df(root_path: str, start: int, stop: Optional[int]) -> pd.DataFrame:
    """
    Get Pandas dataframe of training data, "relevant_articles" is a column with list values, each item
    is the raw HTML text given in the article txt file
    """
    ids = []
    claims = []
    claimants = []
    dates = []
    labels = []
    related_articles = []
    with open(os.path.join(root_path, "train.json")) as json_file:
        data = json.load(json_file)
        for idx, item in enumerate(data):
            if idx < start:
                continue
            if stop and idx > stop:
                break
            parsed_claim = LeadersPrizeClaim(item)
            articles = []
            for related_article in parsed_claim.related_articles:
                articles.append(get_train_article(root_path, related_article.filepath))

            ids.append(parsed_claim.id)
            claims.append(parsed_claim.claim)
            claimants.append(parsed_claim.claimant)
            dates.append(parsed_claim.date)
            labels.append(parsed_claim.label)
            related_articles.append(articles)

    return pd.DataFrame(data={
        "id": ids,
        "claim": claims,
        "claimant": claimants,
        "date": dates,
        "label": labels,
        "related_articles": related_articles
    })


def get_train_article(root_path: str, article_path: str) -> str:
    filepath = os.path.join(root_path, article_path)
    with open(filepath, 'r') as article:
        return article.read()


def train_data_generator(json_path: str):
    with open(json_path) as json_file:
        data = json.load(json_file)
        for idx, item in enumerate(data):
            yield idx, item, LeadersPrizeClaim(item)
