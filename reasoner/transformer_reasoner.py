from typing import List, Dict

from core.models import PipelineClaim
from reasoner.transformers.transformers_sequence_classification import RobertaSequenceClassifier, TransformersInputItem


class TransformerReasoner:

    def __init__(self, sts_scorer: RobertaSequenceClassifier, entailment_scorer: RobertaSequenceClassifier):
        self.sts_scorer = sts_scorer
        self.entailment_scorer = entailment_scorer


    def predict(self, claim: PipelineClaim):

        input_items: Dict[str, TransformersInputItem] = {}  # ID to item

        # Construct input examples, with a generated ID
        for article_id, article in enumerate(claim.articles_for_reasoner):
            for sentence_id, sentence in enumerate(article.sentences_for_reasoner):
                identifier = f"{article_id}-{sentence_id}"
                # Add the ID to the sentence
                sentence.id = identifier
                # Construct input item
                input_item = TransformersInputItem(identifier,
                                                   claim.preprocessed_claim,
                                                   sentence.sentence)
                input_items[identifier] = input_item

        transformer_inputs = list(input_items.values())
        predicted_similarities = self.sts_scorer.predict(transformer_inputs)  # 2 class probabilities
        predicted_entailments = self.entailment_scorer.predict(transformer_inputs)  # 3 class probabilities

        # TODO
