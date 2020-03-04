from typing import List, Dict

from core.models import PipelineClaim
from reasoner.transformers.models import StsSimilarity, Entailment, TransformersInputItem
from reasoner.transformers.transformers_sequence_classification import RobertaSequenceClassifier

# TODO: Extract as config
STS_LIMIT = StsSimilarity.NOT_EQUIV_SHARE_DETAIL


class TransformerReasoner:

    def __init__(self, sts_scorer: RobertaSequenceClassifier, entailment_scorer: RobertaSequenceClassifier):
        self.sts_scorer = sts_scorer
        self.entailment_scorer = entailment_scorer

    def predict(self, claim: PipelineClaim):

        transformers_input_items: List[TransformersInputItem] = []  # Items to predict from

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
                transformers_input_items.append(input_item)

        predicted_similarities = self.sts_scorer.predict(transformers_input_items)  # 1 class regression
        predicted_entailments = self.entailment_scorer.predict(transformers_input_items)  # 3 class logits

        # Iterate, create a dictionary of identifier to predictions
        ids_to_predictions: Dict[str: Dict] = {}
        for (pred_sim, pred_entailment, input_item) in zip(predicted_similarities,
                                                           predicted_entailments,
                                                           transformers_input_items):
            pred_item = {
                "sts_sim": StsSimilarity.from_probabilities(pred_sim),
                "entailment": Entailment.from_probabilities(pred_entailment)
            }
            ids_to_predictions[input_item.uuid] = pred_item

        # Add annotations
        claim_entailment = Entailment.NEUTRAL
        total_claim_entailment = 0.0
        num_considered_articles = 0
        for article in claim.articles_for_reasoner:
            total_article_entailment = 0.0
            num_considered_sentences = 0
            for sentence in article.sentences_for_reasoner:
                # Get predicted item from the dict
                predicted = ids_to_predictions.get(sentence.id, None)
                if not predicted:
                    print("Somehow no predicted item in ID map")
                    continue
                # Get individual predicted items
                sts_sim = predicted.get("sts_sim", None)
                entailment = predicted.get("entailment", None)
                if sts_sim is None or entailment is None:
                    print(f"One or more predicted items not found: {predicted}")
                    continue
                # Annotate the sentence
                sentence.sts_relevance_score = sts_sim
                sentence.entailment_score = entailment

                # Do calculations on aggregate entailment
                if sts_sim > STS_LIMIT:
                    num_considered_sentences += 1
                    total_article_entailment += entailment

            # Calculate entailment for this article
            article_entailment_score = Entailment.NEUTRAL  # Default to neutral
            if num_considered_sentences > 0:
                # Relevant sentences found
                avg_entailment = total_article_entailment / float(num_considered_sentences)
                if avg_entailment > 1.1:
                    article_entailment_score = Entailment.ENTAILMENT
                elif avg_entailment < 0.9:
                    article_entailment_score = Entailment.CONTRADICTION

                # Add to claim processing
                num_considered_articles += 1
                total_claim_entailment += article_entailment_score

            # Annotate
            article.entailment_score = article_entailment_score

        # Calculate claim entailment
        if num_considered_articles > 0:
            # We have found something relevant
            avg_claim_entailment = float(total_claim_entailment) / num_considered_articles
            if avg_claim_entailment > 1.1:
                claim_entailment = Entailment.ENTAILMENT
            elif avg_claim_entailment < 0.9:
                claim_entailment = Entailment.CONTRADICTION

        # Annotate
        claim.submission_label = claim_entailment.value

        return claim
