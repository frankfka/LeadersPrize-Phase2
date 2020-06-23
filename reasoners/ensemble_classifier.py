import os
import pickle
import re

import numpy as np
import pandas as pd
import typing

import scipy
import scipy.spatial
from nltk.corpus import stopwords

from reasoners.paraphrase.paraphrase_classifier import ParaphraseClassifier, ParaphraseResult
from reasoners.rumoreval.rumoreval_classifier import RumorEvalClassifier, RumorResult
from reasoners.common import data_processing, logger_mod
from reasoners.snli.snli_classifier import SnliClassifier, SnliResult


class EnsembleClassifierResult:
    """
    Output of the Ensemble Classifier
    """

    def __init__(self, sent_idx: int, snli: SnliResult, paraphrase: ParaphraseResult, rumor: RumorResult):
        self.sent_idx = sent_idx
        self.snli = snli
        self.paraphrase = paraphrase
        self.rumor = rumor  # Will be none for now


# TODO: Can use this to strip out NEUTRAL predictions
class EnsembleClassifier:
    def __init__(self, root_model_path: str):
        # TODO: Extract config keys
        self.params = {
            'model_type': 'esim',
            'learning_rate': 0.001,
            'keep_rate': 0.5,
            'seq_length': 50,
            'batch_size': 32,
            'word_embedding_dim': 50,
            'hidden_embedding_dim': 50,
            'vocab_path': os.path.join(root_model_path, "ensemble_vocab.p"),
            'embedding_data_path': os.path.join(root_model_path, "glove.6B.50d.txt"),  # TODO: W2V?
            'log_path': root_model_path,
            'ckpt_path': root_model_path
        }
        self.word_indices = self.__load_vocab(self.params["vocab_path"])
        self.loaded_embeddings = data_processing.loadEmbedding_rand(self.params["embedding_data_path"], self.params,
                                                                    self.word_indices)
        self.snli_classifier = self.__init_snli()
        self.paraphrase_classifier = self.__init_paraphrase()
        # self.rumor_classifier = self.__init_rumor()

    def predict_statement_in_contexts(self, statement: str, contexts: typing.List[str]) -> typing.List[typing.Dict]:
        return [self.__predict_statement_in_context(statement, context) for context in contexts]

    def __is_relevant(self, statement, context):
        distance = self.__cosine_distance(statement,
                                          context,
                                          embeddings=self.loaded_embeddings,
                                          word_indices=self.word_indices)
        similarity = 1 - distance
        __is_relevant = similarity >= 0.7  # TODO: Extract as config
        return __is_relevant

    def __predict_statement_in_context(self, statement: str, context: str) -> typing.Dict:
        data = self.__build_statement_context_df(statement, context)
        snli_row = self.__predict_snli(data)
        paraphrase_row = self.__predict_paraphrase(data)
        # rumor_row = self.__predict_rumor(data)
        return {'entailment': snli_row, 'paraphrase': paraphrase_row}

    def __build_statement_context_df(self, statement: str, context: str) -> pd.DataFrame:
        data = {'sentence0': context, 'sentence1': statement}
        data_processing.sentences_to_padded_index_sequences(self.word_indices, self.params, [[data]])
        data = pd.DataFrame([data])
        return data

    def __init_snli(self):
        snli_classifier = SnliClassifier(
            loaded_embeddings=self.loaded_embeddings,
            params=self.params,
            logger=logger_mod.Logger(),
            modname='ensemble_snli',
            emb_train=True,
            loaded_vocab=self.word_indices)
        snli_classifier.restore()
        return snli_classifier

    def __init_paraphrase(self):
        paraphrase_classifier = ParaphraseClassifier(
            loaded_embeddings=self.loaded_embeddings,
            params=self.params,
            logger=logger_mod.Logger(),
            modname='ensemble_paraphrase',
            emb_train=False,
            loaded_vocab=self.word_indices)
        paraphrase_classifier.restore()
        return paraphrase_classifier

    def __init_rumor(self):
        rumor_classifier = RumorEvalClassifier(
            loaded_embeddings=self.loaded_embeddings,
            params=self.params,
            logger=logger_mod.Logger(),
            modname='ensemble_rumoreval')
        rumor_classifier.restore()
        return rumor_classifier

    def __predict_snli(self, data: pd.DataFrame) -> SnliResult:  # hypothesis: str, premise: str):
        try:
            return self.snli_classifier.continue_classify(data)[0]
        except Exception as e:
            print("Error predicting snli")
            print(e)
            return SnliResult.NEUTRAL

    def __predict_paraphrase(self, data: pd.DataFrame) -> ParaphraseResult:  # hypothesis: str, premise: str):
        try:
            return self.paraphrase_classifier.continue_classify(data)[0]
        except Exception as e:
            print("Error predicting paraphrase")
            print(e)
            return ParaphraseResult.NEUTRAL

    # TODO: Fix this
    def __predict_rumor(self, data: pd.DataFrame) -> RumorResult:  # hypothesis: str, premise: str):
        try:
            return self.rumor_classifier.continue_classify(data)[0]
        except Exception as e:
            print("Error predicting rumor")
            print(e)
            return RumorResult.COMMENT

    def __load_vocab(self, vocab_path):
        with open(vocab_path, 'rb') as f:
            dictionary = pickle.load(f)
            return dictionary

    def __cosine_distance(self, s1, s2, embeddings, word_indices):
        vector_1 = np.mean(
            [embeddings[word_indices[word]] for word in self.__preprocess(s1, word_indices=word_indices)],
            axis=0)
        vector_2 = np.mean(
            [embeddings[word_indices[word]] for word in self.__preprocess(s2, word_indices=word_indices)],
            axis=0)
        cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
        return cosine

    def __preprocess(self, raw_text, word_indices):
        # keep only words
        letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

        # convert to lower case and split
        words = letters_only_text.lower().split()

        # remove stopwords
        stopword_set = set(stopwords.words("english"))
        cleaned_words = list(set([w for w in words if w not in stopword_set]))

        filtered_words = [word for word in cleaned_words if word in word_indices.keys()]

        return filtered_words
