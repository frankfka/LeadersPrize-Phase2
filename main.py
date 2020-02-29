import json
import time
from typing import List

from config import METADATA_FILEPATH, PIPELINE_CONFIG, PREDICTIONS_FILEPATH

from core.models import LeadersPrizeClaim, PipelineClaim
from core.pipeline import LeadersPrizePipeline

# Convert input to claim objects
def read_raw_data(filepath: str) -> List[LeadersPrizeClaim]:
    with open(filepath) as json_file:
        data = json.load(json_file)
        return [LeadersPrizeClaim(item) for item in data]


# Writes result to disk
def write_result(predicted: List[PipelineClaim], filepath: str):
    # Converts a predicted claim to a dict for json serialization
    def to_dict(claim: PipelineClaim):
        return {
            claim.original_claim.id: {
                "label": claim.submission_label,
                "related_articles": claim.submission_article_urls,
                "explanation": claim.submission_explanation
            }
        }

    with open(filepath, 'w') as file:
        # Map to dictionary objects
        results = [to_dict(claim) for claim in predicted]
        json.dump(results, file)


def main():
    # Checkpoint
    t = time.time()

    '''
    Read raw input
    '''
    input_claims = read_raw_data(METADATA_FILEPATH)

    # Checkpoint
    print(f"{len(input_claims)} claims loaded in {time.time() - t}s")
    t = time.time()

    '''
    Create pipeline
    '''
    pipeline = LeadersPrizePipeline(config=PIPELINE_CONFIG)
    # Checkpoint
    print(f"Initialized pipeline in {time.time() - t}s")
    t = time.time()

    '''
    Run through pipeline
    '''
    predictions = pipeline.predict(input_claims)

    # Checkpoint
    print(f"Predicted in {time.time() - t}s")
    t = time.time()

    '''
    Write results
    '''
    write_result(predictions, PREDICTIONS_FILEPATH)

    # Checkpoint
    print(f"Results written in {time.time() - t}s")


main()
