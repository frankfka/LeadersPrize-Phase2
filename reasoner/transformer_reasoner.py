from typing import List

from core.models import PipelineClaim
from reasoner.models import TruthRating, TransformersInputItem
from reasoner.transformers.transformers_sequence_classification import RobertaSequenceClassifier, TransformersConfigKeys


class TransformerReasoner:

    def __init__(self, model_path: str, debug=False):
        config = {
            TransformersConfigKeys.BATCH_SIZE: 2,
            TransformersConfigKeys.MAX_SEQ_LEN: 256,  # This is unused
            TransformersConfigKeys.NUM_LABELS: 3,
            TransformersConfigKeys.CONFIG_PATH: model_path,
            TransformersConfigKeys.MODEL_PATH: model_path,
            TransformersConfigKeys.TOK_PATH: model_path
        }
        self.debug = debug
        self.transformer = RobertaSequenceClassifier(config=config)

    def predict(self, claims: List[PipelineClaim]) -> List[TruthRating]:
        text_a_arr: List[str] = []
        text_b_arr: List[str] = []
        for claim in claims:
            text_b_for_transformer = ""
            for sentence in claim.sentences_for_transformer:
                # Sentences from NLTK preserve the period, so no need to inject punctuation
                text_b_for_transformer += sentence.preprocessed_text + " . "
            if not text_b_for_transformer:
                print("Warning: No preprocessed text_b for reasoner")
                # Transformers errors out with empty input - this occurs when we err when searching a query
                # In this case, give some dummy text, but this should never happen
                text_b_for_transformer = "No supporting information provided"
            # Get tokenized
            text_a_arr.append(claim.preprocessed_claim)
            text_b_arr.append(text_b_for_transformer)

        if self.debug:
            print("Predicting using Transformer")
            print(f"Example Text A: {text_a_arr[0]}")
            print(f"Example Text B: {text_b_arr[0]}")

        pred_probabilities = self.transformer.predict(text_a_arr, text_b_arr)
        predictions: List[TruthRating] = []
        for probs in pred_probabilities:
            predictions.append(TruthRating.from_probabilities(probs=probs))

        return predictions
