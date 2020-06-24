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
