"""
Scrap file for processing training data
"""
from experiments.util.experiment_util import get_html_preprocessor, get_text_preprocessor
from experiments.util.train_data_util import get_train_claims_df

import pandas as pd


def export_train_data_pickle(root_path, save_path):
    """
    Saves a dataframe with all the training data (articles in raw HTML) to save_path
    This will create a ~10gb file!
    """
    get_train_claims_df(root_path).to_pickle(save_path)


def preprocess_articles_from_data_pkl(pkl_path, save_path):
    """
    Preprocess related_articles from raw html pickle
    """
    html_preprocessor = get_html_preprocessor()
    text_preprocessor = get_text_preprocessor()

    df = pd.read_pickle(pkl_path).iloc[12000:13510]
    processed_articles_col = []
    for idx, row in df.iterrows():
        html_articles = row["related_articles"]
        processed_articles = []
        for article in html_articles:
            html_processed = html_preprocessor.process(article)
            text_preprocessed = text_preprocessor.process(html_processed.text)
            processed_articles.append(' . '.join(text_preprocessed.bert_sentences))
        processed_articles_col.append(processed_articles)

        if idx % 100 == 0:
            print(idx)
            print(processed_articles[0])

    # Drop HTML articles
    df = df.drop(['related_articles'], axis=1)
    # Add processed articles
    df['related_articles'] = processed_articles_col
    df.to_pickle(save_path)


if __name__ == '__main__':
    preprocess_articles_from_data_pkl(
        "/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/experiments/output/train_data_with_articles.pkl",
        "/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/experiments/output/train_data_with_preprocessed_articles_12000-13510.pkl"
    )