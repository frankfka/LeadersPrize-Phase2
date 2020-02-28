from datetime import datetime
from typing import List

import pandas as pd

from core.models import LeadersPrizeClaim
from core.pipeline import LeadersPrizePipeline
from experiments.util.experiment_util import save_results
from experiments.util.train_data_util import train_data_generator


def test_pipeline():
    raw_claims: List[LeadersPrizeClaim] = []
    process_range = range(500, 520)
    for idx, claim in train_data_generator("/Users/frankjia/Desktop/LeadersPrize/train/train.json"):
        if idx < process_range.start:
            continue
        elif idx >= process_range.stop:
            break
        raw_claims.append(claim)

    start_time = datetime.now()

    # Create pipeline
    pipeline = LeadersPrizePipeline({
        LeadersPrizePipeline.CONFIG_W2V_PATH: "/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/assets/word2vec/GoogleNewsVectors.bin.gz",
        LeadersPrizePipeline.CONFIG_API_KEY: "ff5fdad7-de1f-4a74-bfac-acd42538131f",
        LeadersPrizePipeline.CONFIG_ENDPOINT: "http://lpsa.wrw.org",
        LeadersPrizePipeline.CONFIG_DEBUG: True
    })

    # Run the prediction
    results = pipeline.predict(raw_claims)

    print(f"{len(results)} processed in {datetime.now() - start_time}")

    # Export results
    claims = []
    labels = []
    reasoner_inputs = []
    for res in results:
        claims.append(res.original_claim.claim)
        labels.append(res.original_claim.label)
        reasoner_input = ""
        for article in res.articles_for_reasoner:
            reasoner_input += f"=== NEW Article: {article.relevance} ==="
            for sent in article.sentences_for_reasoner:
                reasoner_input += f"{sent}"
        reasoner_inputs.append(reasoner_input)
    results_df = pd.DataFrame(data={
        "claim": claims,
        "label": labels,
        "reasoner_input": reasoner_inputs
    })

    save_results(results_df, "pipeline_test", "claim_to_bert")


if __name__ == '__main__':
    test_pipeline()
