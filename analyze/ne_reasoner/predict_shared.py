import os

import bs4
import tensorflow.compat.v1 as tf
import numpy as np

from analyze.ne_reasoner import vocab
from analyze.ne_reasoner.models.esim import EsimModel

tf.disable_v2_behavior()


def read_all(path: str):
    file_names = os.listdir(path)
    for file_name in file_names:
        with open(f'{path}/{file_name}', errors='ignore') as f:
            text = f.read()
            yield file_name, text


def strip_newlines(text: str) -> str:
    text = text.replace('\n', '')
    return text


def parse_html(html: str) -> str:
    soup = bs4.BeautifulSoup(html, 'html.parser')

    bad_parents = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head',
        'input',
        'script',
        'style'
    ]

    texts = soup.find_all(text=True)
    texts = [text for text in texts if text.count(' ') > 1]

    bad_chars = [';', '<', '>', '@', '©', '{', '}', 'div', 'email', 'password', ]
    for char in bad_chars:
        texts = [text for text in texts if text.lower().count(char) == 0]
    texts = [text for text in texts if len(text) > 5]
    texts = [text for text in texts
             if text.parent.name not in bad_parents]

    # remove extra whitespace
    texts = [' '.join(text.split()) for text in texts]

    combined_text = ' '.join(texts)
    return combined_text


def load_articles(path='datainput'):
    article_names = []
    articles = []
    for data_names, html in read_all(f'../data/{path}'):
        article = parse_html(html)
        articles.append(article)
        article_names.append(data_names)
    return article_names, articles


class ModelClassifier:
    def __init__(self, loaded_embeddings, parameters, logger, modname, ckpt_path, vocab,
                 num_labels=3, emb_train=False):
        tf.reset_default_graph()
        self.ckpt_path = ckpt_path
        self.logger = logger
        self.num_labels = num_labels
        self.vocab = vocab

        self.modname = modname
        self.loaded_embeddings = loaded_embeddings

        ## Define hyperparameters
        self.learning_rate = parameters["learning_rate"]
        self.display_epoch_freq = 1
        self.display_step_freq = 50
        self.emb_train = emb_train
        self.embedding_dim = parameters["word_embedding_dim"]
        self.dim = parameters["hidden_embedding_dim"]
        self.batch_size = parameters["batch_size"]
        self.keep_rate = parameters["keep_rate"]
        self.sequence_length = parameters["seq_length"]

        self.model = EsimModel(seq_length=self.sequence_length,
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
        best_path = os.path.join(self.ckpt_path, self.modname) + ".ckpt_best"
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
        confidences = logits[1:][0]

        maximum = max(1, max(confidences))
        minimum = min(-1, min(confidences))
        confidences = (confidences - minimum) / (maximum - minimum)  # normalize -1,1 -> 0,1

        return confidences
