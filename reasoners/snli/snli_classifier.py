from enum import IntEnum
from typing import List

import tensorflow.compat.v1 as tf

from reasoners.common.model_classifier import ModelClassifier

tf.disable_v2_behavior()


class SnliResult(IntEnum):
    ENTAILMENT = 0
    NEUTRAL = 1
    CONTRADICTION = 2

    @classmethod
    def from_label(cls, label):
        try:
            return SnliResult(label)
        except Exception as e:
            print(f"Unable to create SnliResult from {label}")
            print(e)
            return SnliResult.NEUTRAL


class SnliClassifier(ModelClassifier):
    def continue_classify(self, examples) -> List[SnliResult]:
        labels = []
        for i, example in examples.iterrows():
            label = super().continue_classify(example.to_frame().T)[0]
            labels.append(SnliResult.from_label(label))
        return labels
