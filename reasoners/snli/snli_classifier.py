
import tensorflow.compat.v1 as tf

from reasoners.common.model_classifier import ModelClassifier

tf.disable_v2_behavior()


class SnliClassifier(ModelClassifier):

    INVERSE_MAP = {
        0: "entailment",
        1: "neutral",
        2: "contradiction"
    }

    def continue_classify(self, examples):
        labels = []
        for i, example in examples.iterrows():
                label = super().continue_classify(example.to_frame().T)[0]
                labels.append(label)
        return labels
