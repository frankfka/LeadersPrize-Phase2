from datetime import datetime
from typing import List, Dict

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from core.models import LeadersPrizeClaim, PipelineClaim
from core.pipeline import LeadersPrizePipeline, PipelineConfigKeys
from experiments.util.datacup_scoring import compute_score_phase2
from experiments.util.experiment_util import save_results
from experiments.util.train_data_util import train_data_generator, get_train_article
from search_client.client import SearchQueryResult


def eval_datacup(true_arr: List[dict], predicted_arr: List[PipelineClaim]):
    pred_dict = {}
    for pred in predicted_arr:
        pred_dict[pred.submission_id] = {
            "label": pred.submission_label,
            "related_articles": pred.submission_article_urls,
            "explanation": pred.submission_explanation
        }
    true_dict = {}
    for true_item in true_arr:
        true_dict[true_item["id"]] = true_item
    score_result = compute_score_phase2(pred_dict, true_dict)
    score = score_result["score"]
    print(f"Datacup Score: {score}")


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
    PipelineConfigKeys.DEBUG_MODE: True,
    PipelineConfigKeys.RETRIEVE_ARTICLES: True,
}
PROCESS_RANGE = range(0, 600)
TRAIN_DATA_PATH = "/Users/frankjia/Desktop/LeadersPrize/train/"


def test_pipeline(process_range: range, config: Dict, train_data_path: str):
    raw_claims: List[LeadersPrizeClaim] = []
    raw_claim_dicts: List[dict] = []
    init_articles = not config.get(PipelineConfigKeys.RETRIEVE_ARTICLES, True)
    if init_articles:
        print("Reading articles from training data. Will not call search client")
    for idx, claim_dict, claim in train_data_generator(train_data_path + "trial_combined_data_long.json"):
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
        raw_claim_dicts.append(claim_dict)

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
    supporting_article_urls = []
    explanations = []
    for res in results:
        claims.append(res.preprocessed_claim)
        labels.append(res.original_claim.label)
        reasoner_input = ""
        for idx, sent in enumerate(res.sentences_for_transformer):
            if idx == 10:
                break
            reasoner_input += " " + sent.preprocessed_text
        reasoner_inputs.append(reasoner_input)
        pred_labels.append(res.submission_label)
        supporting_article_urls.append(", ".join(res.submission_article_urls.values()))
        explanations.append(res.submission_explanation)
    results_df = pd.DataFrame(data={
        "claim": claims,
        "label": labels,
        "reasoner_input": reasoner_inputs,
        "predicted": pred_labels,
        "article_urls": supporting_article_urls,
        "explanation": explanations
    })

    # Get accuracies
    eval_predictions(labels, pred_labels)
    # Get datacup score
    eval_datacup(raw_claim_dicts, results)

    save_results(results_df, "pipeline_test", "full_pipeline")


if __name__ == '__main__':
    test_pipeline(PROCESS_RANGE, PIPELINE_CONFIG, TRAIN_DATA_PATH)
