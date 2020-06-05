import os

import tensorflow.compat.v1 as tf
import numpy as np

from reasoners.common import esim

tf.disable_v2_behavior()


class ModelClassifier:
    def __init__(self, loaded_vocab, loaded_embeddings, params, logger, modname,
                 num_labels=3, emb_train=False):
        tf.reset_default_graph()
        self.params = params
        self.logger = logger
        self.num_labels = num_labels
        self.vocab = loaded_vocab

        self.modname = modname
        self.loaded_embeddings = loaded_embeddings

        ## Define hyperparameters
        self.learning_rate = self.params["learning_rate"]
        self.display_epoch_freq = 1
        self.display_step_freq = 50
        self.emb_train = emb_train
        self.embedding_dim = self.params["word_embedding_dim"]
        self.dim = self.params["hidden_embedding_dim"]
        self.batch_size = self.params["batch_size"]
        self.keep_rate = self.params["keep_rate"]
        self.sequence_length = self.params["seq_length"]

        self.model = esim.EsimModel(
            seq_length=self.sequence_length,
            emb_dim=self.embedding_dim,
            hidden_dim=self.dim,
            embeddings=loaded_embeddings,
            emb_train=self.emb_train,
            num_labels=self.num_labels
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
        premise_vectors = np.vstack(batch['sentence0_index_sequence'])
        hypothesis_vectors = np.vstack(batch['sentence1_index_sequence'])
        return premise_vectors, hypothesis_vectors

    def restore(self):
        best_path = os.path.join(self.params["ckpt_path"], self.modname) + ".ckpt_best"
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.saver.restore(self.sess, best_path)
        self.logger.Log("Model restored from file: %s" % best_path)

    def classify(self, examples):
        self.restore()
        return self.continue_classify(examples)

    def continue_classify(self, examples):
        logits = np.empty(self.num_labels)
        minibatch_premise_vectors, minibatch_hypothesis_vectors = self.get_minibatch(examples, 0, len(examples))
        feed_dict = {self.model.premise_x: minibatch_premise_vectors,
                     self.model.hypothesis_x: minibatch_hypothesis_vectors,
                     self.model.keep_rate_ph: 1.0}
        logit = self.sess.run(self.model.logits, feed_dict)
        logits = np.vstack([logits, logit])

        return np.argmax(logits[1:], axis=1)
