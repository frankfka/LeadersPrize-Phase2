"""
Export Preprocessed Training Data for BERT
"""
from typing import List

import pandas as pd

from core.models import PipelineSentence
from experiments.util.experiment_util import get_word2vec_relevance_scorer
from experiments.util.train_data_util import get_preprocessed_train_data
from preprocess.text_util import tokenize_by_sentence


def combine_bert_trained_dfs(paths: List[str], output_path: str):
    """
    Combine the mini-batches of exported training BERT data
    """
    dfs = []
    for path in paths:
        dfs.append(pd.read_pickle(path))
    df: pd.DataFrame = pd.concat(dfs)
    df.to_pickle(output_path)


def export_bert_train_data(preprocessed_pickle_path: str, sentence_extraction_window: int, output_path: str):
    # Get dependencies
    relevance_scorer = get_word2vec_relevance_scorer()

    # Read the preprocessed pickle dataframe
    preprocessed_df = get_preprocessed_train_data(preprocessed_pickle_path)

    # Build text_1, text_2, labels
    claims = []
    extracted_info = []
    labels = []

    for idx, row in preprocessed_df.iterrows():
        claim: str = row["claimant"] + " " + row["claim"]
        label: int = int(row["label"])
        related_articles: List[str] = row["related_articles"]
        pipeline_sentences: List[PipelineSentence] = []

        # For text_2, need to get relevance, order
        # Run relevance scorer - put all the articles toeether
        for article in related_articles:
            article_sentences = tokenize_by_sentence(article)
            for sent in article_sentences:
                relevance = relevance_scorer.get_relevance(claim, sent)
                pipeline_sent = PipelineSentence()
                pipeline_sent.text = sent
                pipeline_sent.relevance = relevance
                pipeline_sentences.append(pipeline_sent)

        pipeline_sentences.sort(key=lambda sentence: sentence.relevance, reverse=True)

        claims.append(claim)
        labels.append(label)
        # To save space, limit how many words we add to the file
        num_words = 0
        extracted_txt = ""
        for sent in pipeline_sentences:
            if num_words > 1000:
                break
            extracted_txt += sent.text + " "
            num_words += len(sent.text.split())
        extracted_info.append(extracted_txt)

        if idx % 100 == 0:
            print(idx)
            print(claim)
            print(label)
            print(extracted_txt)
            print("\n")

    # Export results
    export_df = pd.DataFrame(data={"text_1": claims, "text_2": extracted_info, "label": labels})
    export_df.to_pickle(output_path)


if __name__ == '__main__':
    # export_bert_train_data(
    #     "/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/experiments/output/train_data_preprocessed_16000_END.pkl",
    #     1,
    #     "/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/experiments/output/bert_train_data_0window_16000_END.pkl"
    # )
    combine_bert_trained_dfs([
        "/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/experiments/output/bert_train_data_0window_0_2000.pkl",
        "/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/experiments/output/bert_train_data_0window_2000_6000.pkl",
        "/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/experiments/output/bert_train_data_0window_6000_10000.pkl",
        "/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/experiments/output/bert_train_data_0window_10000_16000.pkl",
        "/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/experiments/output/bert_train_data_0window_16000_END.pkl"
    ],
    "/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/experiments/output/bert_train_data_w2v_0window_claimant.pkl"
    )