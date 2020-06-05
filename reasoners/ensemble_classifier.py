import pickle
import re

import numpy as np
import pandas as pd
import typing

import scipy
import scipy.spatial
from nltk.corpus import stopwords

from reasoners.paraphrase.paraphrase_classifier import ParaphraseClassifier
from reasoners.rumoreval.rumoreval_classifier import RumorEvalClassifier
from reasoners.common import data_processing, logger_mod
from reasoners.snli.snli_classifier import SnliClassifier

PARAMS = {
    'model_type': 'esim',
    'learning_rate': 0.001,
    'keep_rate': 0.5,
    'seq_length': 50,
    'batch_size': 32,
    'word_embedding_dim': 50,
    'hidden_embedding_dim': 50,
    'vocab_path': "/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/assets/reasoner/ensemble_vocab.p",
    'embedding_data_path': '/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/assets/reasoner/glove.6B.50d.txt',
    'log_path': '/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/assets/reasoner',
    'ckpt_path': '/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/assets/reasoner'
}


class EnsembleClassifier:
    def __init__(self):
        self.params = PARAMS
        self.data_root_path = '/Users/frankjia/Desktop/LeadersPrize/LeadersPrize-Phase2/assets/reasoner'
        self.word_indices = self.load_vocab(self.params["vocab_path"])
        self.loaded_embeddings = data_processing.loadEmbedding_rand(PARAMS["embedding_data_path"], self.params, self.word_indices)
        self.snli_classifier = self.init_snli()
        self.paraphrase_classifier = self.init_paraphrase()
        self.rumor_classifier = self.init_rumor()

    def predict_statement_in_contexts(self, statement: str, contexts: typing.List[str]):
        lines = []
        predictions = []
        for i, context in enumerate(contexts):
            is_relevant = self.is_relevant(statement, context)
            if is_relevant:
                prediction = self.predict_statement_in_context(statement, context)
                lines.append(i)
                predictions.append(prediction)
        return lines, predictions

    def is_relevant(self, statement, context):
        distance = self.cosine_distance(statement,
                                        context,
                                        embeddings=self.loaded_embeddings,
                                        word_indices=self.word_indices)
        similarity = 1 - distance
        is_relevant = similarity >= 0.7
        return is_relevant

    def predict_statement_in_context(self, statement: str, context: str):
        data = self.build_statement_context_df(statement, context)
        snli_row = self.predict_snli(data)
        paraphrase_row = self.predict_paraphrase(data)
        rumor_row = self.predict_rumor(data)
        return {'entailment': snli_row, 'paraphrase': paraphrase_row, 'rumor': rumor_row}

    def build_statement_context_df(self, statement: str, context: str) -> pd.DataFrame:
        data = {'sentence0': context, 'sentence1': statement}
        data_processing.sentences_to_padded_index_sequences(self.word_indices, self.params, [[data]])
        data = pd.DataFrame([data])
        return data

    def init_snli(self):
        snli_classifier = SnliClassifier(
            loaded_embeddings=self.loaded_embeddings,
            params=self.params,
            logger=logger_mod.Logger(),
            modname='ensemble_snli',
            emb_train=True,
            loaded_vocab=self.word_indices)
        snli_classifier.restore()
        return snli_classifier

    def init_paraphrase(self):
        paraphrase_classifier = ParaphraseClassifier(
            loaded_embeddings=self.loaded_embeddings,
            params=self.params,
            logger=logger_mod.Logger(),
            modname='ensemble_paraphrase',
            emb_train=False,
            loaded_vocab=self.word_indices)
        paraphrase_classifier.restore()
        return paraphrase_classifier

    def init_rumor(self):
        rumor_classifier = RumorEvalClassifier(
            loaded_embeddings=self.loaded_embeddings,
            params=self.params,
            logger=logger_mod.Logger(),
            modname='ensemble_rumoreval')
        rumor_classifier.restore()
        return rumor_classifier

    def predict_snli(self, data: pd.DataFrame):  # hypothesis: str, premise: str):
        try:
            result = self.snli_classifier.continue_classify(data)[0]
            label = SnliClassifier.INVERSE_MAP[result]
            return label
        except IndexError:
            return None

    def predict_paraphrase(self, data: pd.DataFrame):  # hypothesis: str, premise: str):
        result = self.paraphrase_classifier.continue_classify(data)[0]
        label = ParaphraseClassifier.INVERSE_MAP[result]
        return label

    def predict_rumor(self, data: pd.DataFrame):  # hypothesis: str, premise: str):
        result = self.paraphrase_classifier.continue_classify(data)[0]
        label = RumorEvalClassifier.INVERSE_MAP[result]
        return label

    def load_vocab(self, vocab_path):
        with open(vocab_path, 'rb') as f:
            dictionary = pickle.load(f)
            return dictionary

    def cosine_distance(self, s1, s2, embeddings, word_indices):
        vector_1 = np.mean([embeddings[word_indices[word]] for word in self.preprocess(s1, word_indices=word_indices)],
                           axis=0)
        vector_2 = np.mean([embeddings[word_indices[word]] for word in self.preprocess(s2, word_indices=word_indices)],
                           axis=0)
        cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
        return cosine

    def preprocess(self, raw_text, word_indices):
        # keep only words
        letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

        # convert to lower case and split
        words = letters_only_text.lower().split()

        # remove stopwords
        stopword_set = set(stopwords.words("english"))
        cleaned_words = list(set([w for w in words if w not in stopword_set]))

        filtered_words = [word for word in cleaned_words if word in word_indices.keys()]

        return filtered_words
