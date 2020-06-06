from enum import IntEnum
from typing import List

from reasoners.common.model_classifier import ModelClassifier
import tensorflow.compat.v1 as tf


tf.disable_v2_behavior()


class ParaphraseResult(IntEnum):
    PARAPHRASE = 1
    NEUTRAL = 0

    @classmethod
    def from_label(cls, label):
        try:
            return ParaphraseResult(label)
        except Exception as e:
            print(f"Unable to create ParaphraseResult from {label}")
            print(e)
            return ParaphraseResult.NEUTRAL


class ParaphraseClassifier(ModelClassifier):
    def continue_classify(self, examples) -> List[ParaphraseResult]:
        labels = []
        for i, example in examples.iterrows():
            label = super().continue_classify(example.to_frame().T)[0]
            labels.append(ParaphraseResult.from_label(label))
        return labels
