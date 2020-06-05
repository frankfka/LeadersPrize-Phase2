from reasoners.common.model_classifier import ModelClassifier
import tensorflow.compat.v1 as tf


tf.disable_v2_behavior()


class ParaphraseClassifier(ModelClassifier):
    INVERSE_MAP = {
        1: "paraphrase",
        0: "neutral"
    }

    def continue_classify(self, examples):
        labels = []
        for i, example in examples.iterrows():
            label = super().continue_classify(example.to_frame().T)[0]
            labels.append(label)
        return labels
