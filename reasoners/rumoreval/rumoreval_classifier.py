import os
from enum import IntEnum
from typing import List

import numpy as np

import tensorflow.compat.v1 as tf

from reasoners.common.esim import EsimModel
from reasoners.common.model_classifier import ModelClassifier

tf.disable_v2_behavior()


class RumorResult(IntEnum):
    COMMENT = 0
    SUPPORT = 1
    QUERY = 2
    DENY = 3

    @classmethod
    def from_label(cls, label):
        try:
            return RumorResult(label)
        except Exception as e:
            print(f"Unable to create ParaphraseResult from {label}")
            print(e)
            return RumorResult.NEUTRAL


class RumorEvalClassifier(ModelClassifier):

    def __init__(self, loaded_embeddings, params, logger, modname):
        tf.reset_default_graph()
        self.params = params
        self.logger = logger
        self.modname = modname

        ## Define hyperparameters
        self.learning_rate = self.params["learning_rate"]
        self.display_epoch_freq = 1
        self.display_step_freq = 50
        self.embedding_dim = self.params["word_embedding_dim"]
        self.dim = self.params["hidden_embedding_dim"]
        self.batch_size = self.params["batch_size"]
        self.keep_rate = self.params["keep_rate"]
        self.sequence_length = self.params["seq_length"]
        self.num_labels = 4

        self.model = EsimModel(seq_length=self.sequence_length, emb_dim=self.embedding_dim,
                               hidden_dim=self.dim,
                               embeddings=loaded_embeddings, emb_train=False,
                               num_labels=4
                               )

        # Perform gradient descent with Adam
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999).minimize(
            self.model.total_cost)

        # tf things: initialize variables and create placeholder for session
        self.logger.Log("Initializing variables")
        self.init = tf.global_variables_initializer()
        self.sess = None
        self.saver = tf.train.Saver()

    def get_minibatch(self, dataset, start_index, end_index):
        batch = dataset[start_index: end_index]
        vectors = np.vstack(batch['text_index_sequence'])
        return vectors

    def continue_classify(self, examples) -> List[RumorResult]:
        logits = np.empty(4)
        minibatch_vectors = self.get_minibatch(examples, 0, len(examples))
        feed_dict = {self.model.premise_x: minibatch_vectors,
                     self.model.hypothesis_x: minibatch_vectors,
                     self.model.keep_rate_ph: 1.0}
        logit = self.sess.run(self.model.logits, feed_dict)
        logits = np.vstack([logits, logit])

        argmax_logits = np.argmax(logits[1:], axis=1)
        return [RumorResult.from_label(r) for r in argmax_logits]
