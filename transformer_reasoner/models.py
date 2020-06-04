from enum import IntEnum
import numpy as np
import math


class TransformersInputItem:

    def __init__(self, uuid, text_a, text_b, label=None):
        self.uuid = uuid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class TruthRating(IntEnum):
    TRUE = 2
    NEUTRAL = 1
    FALSE = 0

    @classmethod
    def from_probabilities(cls, probs):
        max_idx = np.argmax(probs)
        return TruthRating(max_idx)


class Entailment(IntEnum):
    ENTAILMENT = 2
    NEUTRAL = 1
    CONTRADICTION = 0

    @classmethod
    def from_probabilities(cls, probs):
        max_idx = np.argmax(probs)
        return Entailment(max_idx)


class StsSimilarity(IntEnum):
    EQUIV = 5
    MOSTLY_EQUIV = 4
    ROUGH_EQUIV = 3
    NOT_EQUIV_SHARE_DETAIL = 2
    NOT_EQUIV_SAME_TOPIC = 1
    DISSIMILAR = 0

    @classmethod
    def from_probabilities(cls, pred):
        # Round up to nearest int
        # TODO: make this a float instead
        return StsSimilarity(math.ceil(pred[0]))
