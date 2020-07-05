import tensorflow.compat.v1 as tf

from analyze.ne_reasoner import predict_shared

tf.disable_v2_behavior()


class SnliClassifier(predict_shared.ModelClassifier):
    def continue_classify(self, examples):
        confidences = []
        for i, example in examples.iterrows():
            confidence = super().continue_classify(example.to_frame().T)
            confidences.append(confidence)
        avg_confidences = sum(confidences) / len(confidences)
        return avg_confidences

