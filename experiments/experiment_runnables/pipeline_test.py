from datetime import datetime
from typing import List, Dict

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from core.models import LeadersPrizeClaim
from core.pipeline import LeadersPrizePipeline, PipelineConfigKeys
from experiments.util.experiment_util import save_results
from experiments.util.train_data_util import train_data_generator, get_train_article
from search_client.client import SearchQueryResult


def eval_predictions(y_true, y_pred):
    # Plot confusion matrix
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1_score_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_score_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_score_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score (Macro): {f1_score_macro}")
    print(f"F1 Score (Micro): {f1_score_micro}")
    print(f"F1 Score (Weighted): {f1_score_weighted}")


PROJ_ROOT = "/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/"
PIPELINE_CONFIG = {
    PipelineConfigKeys.W2V_PATH: f"{PROJ_ROOT}assets/word2vec/GoogleNewsVectors.bin.gz",
    PipelineConfigKeys.API_KEY: "ff5fdad7-de1f-4a74-bfac-acd42538131f",
    PipelineConfigKeys.ENDPOINT: "http://lpsa.wrw.org",
    PipelineConfigKeys.TRANSFORMER_PATH: f"{PROJ_ROOT}assets/roberta_reasoner/",
    PipelineConfigKeys.NE_REASONER_PATH: f"{PROJ_ROOT}assets/ne_reasoner/",
    PipelineConfigKeys.DEBUG_MODE: True,
    PipelineConfigKeys.RETRIEVE_ARTICLES: True,
}
PROCESS_RANGE = range(0, 100)
TRAIN_DATA_PATH = "/Users/frankjia/Desktop/LeadersPrize/train/"


def test_pipeline(process_range: range, config: Dict, train_data_path: str):
    raw_claims: List[LeadersPrizeClaim] = []
    init_articles = not config.get(PipelineConfigKeys.RETRIEVE_ARTICLES, True)
    if init_articles:
        print("Reading articles from training data. Will not call search client")
    for idx, claim in train_data_generator(train_data_path + "trial_combined_data.json"):
        if idx < process_range.start:
            continue
        elif idx >= process_range.stop:
            break
        # Add the articles if we're not retrieving from search client
        if init_articles:
            articles: List[SearchQueryResult] = []
            for related_article in claim.related_articles:
                article_html = get_train_article(train_data_path, related_article.filepath)
                articles.append(
                    SearchQueryResult(content=article_html, url=related_article.url)
                )
            claim.mock_search_results = articles
        raw_claims.append(claim)

    start_time = datetime.now()

    # Create pipeline
    pipeline = LeadersPrizePipeline(config)

    # Run the prediction
    results = pipeline.predict(raw_claims)

    print(f"{len(results)} processed in {datetime.now() - start_time}")

    # Export results
    claims = []
    labels = []
    reasoner_inputs = []
    pred_labels = []
    for res in results:
        claims.append(res.preprocessed_claim)
        labels.append(res.original_claim.label)
        reasoner_input = ""
        for idx, sent in enumerate(res.sentences_for_transformer):
            if idx == 10:
                break
            reasoner_input += " . " + sent.preprocessed_text
        reasoner_inputs.append(reasoner_input)
        pred_labels.append(res.submission_label)
    results_df = pd.DataFrame(data={
        "claim": claims,
        "label": labels,
        "predicted": pred_labels,
        "reasoner_input": reasoner_inputs,
    })

    eval_predictions(labels, pred_labels)

    save_results(results_df, "pipeline_test", "claim_to_preprocessed_train_articles")


if __name__ == '__main__':
    test_pipeline(PROCESS_RANGE, PIPELINE_CONFIG, TRAIN_DATA_PATH)
