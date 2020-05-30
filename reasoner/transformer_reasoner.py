from typing import List, Dict

from core.models import PipelineClaim
from reasoner.models import TruthRating, TransformersInputItem
from reasoner.transformers.transformers_sequence_classification import RobertaSequenceClassifier, TransformersConfigKeys


class TransformerReasoner:

    def __init__(self, model_path: str, debug=False):
        config = {
            TransformersConfigKeys.BATCH_SIZE: 16,
            TransformersConfigKeys.MAX_SEQ_LEN: 256,
            TransformersConfigKeys.NUM_LABELS: 3,
            TransformersConfigKeys.CONFIG_PATH: model_path,
            TransformersConfigKeys.MODEL_PATH: model_path,
            TransformersConfigKeys.TOK_PATH: model_path
        }
        self.debug = debug
        self.transformer = RobertaSequenceClassifier(config=config)

    def predict(self, claims: List[PipelineClaim]) -> List[TruthRating]:
        input_items: List[TransformersInputItem] = []
        for claim in claims:
            if not claim.preprocessed_text_b_for_reasoner:
                print("Warning: No preprocessed text_b for reasoner")
                # Transformers errors out with empty input - this occurs when we err when searching a query
                # In this case, give some dummy text, but TODO: figure this out
                claim.preprocessed_text_b_for_reasoner = "No supporting info provided"
            # Get tokenized
            input_item = TransformersInputItem(claim.original_claim.id,
                                               claim.preprocessed_claim,
                                               claim.preprocessed_text_b_for_reasoner)
            input_items.append(input_item)

        pred_probabilities = self.transformer.predict(input_items=input_items, debug=self.debug)
        predictions: List[TruthRating] = []
        for probs in pred_probabilities:
            predictions.append(TruthRating.from_probabilities(probs=probs))

        return predictions
