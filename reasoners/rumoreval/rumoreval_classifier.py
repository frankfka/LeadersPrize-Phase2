import os

import numpy as np

import tensorflow.compat.v1 as tf

from reasoners.common.esim import EsimModel
from reasoners.common.model_classifier import ModelClassifier

tf.disable_v2_behavior()


class RumorEvalClassifier(ModelClassifier):
    INVERSE_MAP = {
        0: 'comment',
        1: 'support',
        2: 'query',
        3: 'deny'
    }

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

    def classify(self, examples):
        # This classifies a list of examples
        best_path = os.path.join(self.params["ckpt_path"], self.modname) + ".ckpt_best"
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.saver.restore(self.sess, best_path)
        self.logger.Log("Model restored from file: %s" % best_path)

        logits = np.empty(4)
        minibatch_vectors = self.get_minibatch(examples, 0, len(examples))
        feed_dict = {self.model.premise_x: minibatch_vectors,
                     self.model.hypothesis_x: minibatch_vectors,
                     self.model.keep_rate_ph: 1.0}
        logit = self.sess.run(self.model.logits, feed_dict)
        logits = np.vstack([logits, logit])

        return np.argmax(logits[1:], axis=1)
