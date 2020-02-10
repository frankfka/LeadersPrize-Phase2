"""
Test extraction of sentences (i.e. most relevant info ordered first)
"""
from typing import List

import pandas as pd

from core.models import PipelineSentence
from experiments.util.experiment_util import get_word2vec_relevance_scorer, get_doc_sentence_extractor, save_results
from experiments.util.train_data_util import get_preprocessed_train_data
from preprocess.text_util import tokenize_by_sentence


def test_extraction_from_preprocessed_train_data(processed_pkl_path: str):
    # Get relevance scorer
    relevance_scorer = get_word2vec_relevance_scorer()

    # Get extractor
    sentence_extractor = get_doc_sentence_extractor()

    # Read the train data
    preprocessed_df = get_preprocessed_train_data(processed_pkl_path)

    claims = []
    extracted_info = []

    for idx, row in preprocessed_df.iterrows():
        claim: str = row["claim"]
        related_articles: List[str] = row["related_articles"]
        pipeline_sentences: List[PipelineSentence] = []

        # Run relevance scorer - put all the articles toeether
        for article in related_articles:
            article_sentences = tokenize_by_sentence(article)
            for sent in article_sentences:
                relevance = relevance_scorer.get_relevance(claim, sent)
                pipeline_sent = PipelineSentence(sent)
                pipeline_sent.relevance = relevance
                pipeline_sentences.append(pipeline_sent)

        # Extract sentences from annotated items
        extracted_sentences = sentence_extractor.extract(pipeline_sentences, window=1)

        claims.append(claim)
        extracted_info.append(' . '.join([x.sentence for x in extracted_sentences]))

    # Export results
    export_df = pd.DataFrame(data={"claim": claims, "extracted": extracted_info})
    save_results(export_df, "relevant_information_extractor", "w2v_windowsize1")


if __name__ == '__main__':
    test_extraction_from_preprocessed_train_data("/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/experiments/output/train_data_with_preprocessed_articles_0-2000.pkl")