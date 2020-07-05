"""
Script to generate a CSV file of predictions on the test data.
"""

import tensorflow.compat.v1 as tf
import numpy as np

from analyze.ne_reasoner import predict_shared

tf.disable_v2_behavior()


class ParaphraseClassifier(predict_shared.ModelClassifier):
    # def __init__(self, loaded_embeddings, processing, logger, modname, emb_train=False):
    #     super().__init__(loaded_embeddings, processing, logger, modname, num_labels=2, emb_train=emb_train, )

    def continue_classify(self, examples):
        confidences = []
        for i, example in examples.iterrows():
            # confidence = super().continue_classify(example.to_frame().T) #todo should be restored to this
            confidence = self._continue_classify(example.to_frame().T)
            confidences.append(confidence)
        avg_confidences = sum(confidences) / len(confidences)
        return avg_confidences

    def _continue_classify(self, examples):
        # todo: this is just a way to change the num labels without breaking the classifier. Just ignores the
        #   third label. This should be removed once the model is retrained with only 2 labels as it should be.
        logits = np.empty(self.num_labels)
        minibatch_premise_vectors, minibatch_hypothesis_vectors = self.get_minibatch(examples, 0, len(examples))
        feed_dict = {self.model.premise_x: minibatch_premise_vectors,
                     self.model.hypothesis_x: minibatch_hypothesis_vectors,
                     self.model.keep_rate_ph: 1.0}
        logit = self.sess.run(self.model.logits, feed_dict)
        logits = np.vstack([logits, logit])
        confidences = logits[1:][0]

        #todo only difference here
        confidences = confidences[:-1]

        maximum = max(1, max(confidences))
        minimum = min(-1, min(confidences))
        normalized = (confidences - minimum) / (maximum-minimum)  # normalize -1,1 -> 0,1

        # confidences = (confidences - min(confidences)) / (max(confidences) - min(confidences))  # normalize -1,1 -> 0,1
        #
        # try:
        #     confidences = confidences / sum(confidences)  # make sum to 1
        # except ZeroDivisionError:
        #     pass

        return normalized
