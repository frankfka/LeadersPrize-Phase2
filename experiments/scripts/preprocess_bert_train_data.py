"""
Scrap file for processing training data
"""
import pandas as pd

from experiments.util.experiment_util import get_html_preprocessor, get_text_preprocessor
from experiments.util.train_data_util import get_train_claims_df
from preprocess import text_util


def export_train_data_pickle(root_path, save_path):
    """
    Saves a dataframe with all the training data (articles in raw HTML) to save_path
    This will create a ~10gb file!
    """
    get_train_claims_df(root_path, start=12000, stop=None).to_pickle(save_path)


def preprocess_articles_from_data_pkl(pkl_path, save_path):
    """
    Preprocess related_articles from raw html pickle
    """
    html_preprocessor = get_html_preprocessor()
    text_preprocessor = get_text_preprocessor()

    df = pd.read_pickle(pkl_path)
    processed_articles_col = []
    for idx, row in df.iterrows():
        print_ex = idx % 100 == 0
        if print_ex:
            print(idx)

        html_articles = row["related_articles"]
        processed_articles = []
        for idx, article in enumerate(html_articles):
            html_processed = html_preprocessor.process(article)
            if not html_processed.text:
                row_id = row["id"]
                print(f"No text found in one article for claim ID {row_id}")
            sentences = text_util.tokenize_by_sentence(html_processed.text)
            text_preprocessed_sentences = text_preprocessor.process_sentences(sentences)
            processed_articles.append(' '.join(text_preprocessed_sentences))

            if print_ex and idx == 0:
                print("== HTML Preprocessed ==")
                print(html_processed.text)
                print("== Text Preprocessed ==")
                print(' '.join(text_preprocessed_sentences))
                print("\n")

        processed_articles_col.append(processed_articles)

    # Drop HTML articles
    df = df.drop(['related_articles'], axis=1)
    # Add processed articles
    df['related_articles'] = processed_articles_col
    df.to_pickle(save_path)


if __name__ == '__main__':
    preprocess_articles_from_data_pkl(
        "/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/experiments/output/train_data_df_12000_END.pkl",
        "/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/experiments/output/train_data_preprocessed_12000_END.pkl"
    )
    # export_train_data_pickle("/Users/frankjia/Desktop/LeadersPrize/train_phase2_val/",
    #                          "/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/experiments/output/train_data_df_phase2_val.pkl")
