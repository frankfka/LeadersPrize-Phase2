import functools
import numpy as np

import pandas as pd
import typing
import typing as t
import textacy

from analyze.ne_reasoner import vocab, data_processing, term_scoring, similarity_mod, predict_snli, snli_util, \
    logger_mod, reasoner_models
from analyze.ne_reasoner.propositions.main import init_weights, is_proposition

SNLI_LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": 0
}

SNLI_INVERSE_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}
SNLI_PARAMS = {
    'model_type': 'esim',
    # 'learning_rate': 0.0004,
    'learning_rate': 0.001,
    'keep_rate': 0.5,
    'seq_length': 50,
    'batch_size': 32,
    'word_embedding_dim': 50,
    'hidden_embedding_dim': 50,
}

class EnsembleClassifier:
    term_column_names = [
        'evidence',
        'asserting',
        'hedging',
        'questioning',
        'disagreeing',
        'stancing',
        'negative',
        'fakeness', ]

    def __init__(self, vocab_path: str, glove_emb_path: str, cvm_path: str, ckpt_path: str):
        self.cvm_path = cvm_path
        self.word_indices = vocab.load_dictionary(vocab_path)
        self.loaded_embeddings = data_processing.loadEmbedding_rand(
            glove_emb_path,
            self.word_indices
        )
        snli_classifier = predict_snli.SnliClassifier(
            loaded_embeddings=self.loaded_embeddings,
            parameters=SNLI_PARAMS,
            logger=logger_mod.Logger(),
            modname='ensemble_snli',
            ckpt_path=ckpt_path,
            vocab=self.word_indices,
            emb_train=True
        )
        snli_classifier.restore()
        self.snli_classifier = snli_classifier

        # todo move to separate classes

    def predict_for_claim(self, statement: str, contexts: typing.List[str]) -> (reasoner_models.Prediction, typing.List[int], typing.List[dict]):
        relevant_sent_idxs, predicted_attributes = self.predict_statement_in_contexts(statement, contexts)
        false_magnitudes = []
        neutral_magnitudes = []
        true_magnitudes = []
        for pred, sent_idx in zip(predicted_attributes, relevant_sent_idxs):
            sent = contexts[sent_idx]
            analysis_class = reasoner_models.BeliefAnalysis.from_dict(pred)
            false_magnitudes.append(analysis_class.false_magnitude)
            neutral_magnitudes.append(analysis_class.neutral_magnitude)
            true_magnitudes.append(analysis_class.true_magnitude)

        # TODO: if empty
        false_pred_mean = np.mean(false_magnitudes)
        neutral_pred_mean = np.mean(neutral_magnitudes)
        true_pred_mean = np.mean(true_magnitudes)
        max_idx = np.argmax([false_pred_mean, neutral_pred_mean, true_pred_mean])

        # TODO: Proper return
        return [false_pred_mean, neutral_pred_mean, true_pred_mean], relevant_sent_idxs, predicted_attributes


    def predict_statement_in_contexts(self, statement: str, contexts: typing.List[str]):
        lines = []
        predictions = []
        for i, context in enumerate(contexts):
            if self.is_relevant(statement, context) and ensemble_is_proposition(context):
                prediction = self.predict_statement_in_context(statement, context)
                lines.append(i)
                predictions.append(prediction)
        return lines, predictions

    def predict_statement_in_context(self, statement: str, context: str):
        predictions = {}

        data = self.build_statement_context_df(statement, context)
        predictions.update(self.predict_snli(data))

        predictions['paraphrase'] = float(self.is_paraphrase(statement, context))

        terms = {column_name: term_scoring.get_terms(column_name, cvm_path=self.cvm_path)
                 for column_name in self.term_column_names}
        term_predictions = {column_name: self.predict_terms(data, terms[column_name])
                            for column_name in self.term_column_names}
        unskewed = {column_name: term_scoring.unskew_term_predictions(column_name, term_predictions[column_name])
                    for column_name in self.term_column_names}
        predictions.update(unskewed)

        # TODO: Figure out similarity
        predictions['relevance'] = float(self.__get_similarity(statement, context))
        return predictions

    def is_relevant(self, statement, context) -> bool:
        return self._check_similarity(statement, context, 0.65)

    def is_paraphrase(self, statement, context) -> bool:
        return self._check_similarity(statement, context, 0.80)

    def _check_similarity(self, statement, context, similarity_threshold) -> bool:
        return self.__get_similarity(statement, context) >= similarity_threshold

    def __get_similarity(self, statement, context) -> float:
        # TODO: Optimize
        distance = similarity_mod.cosine_distance(statement,
                                                  context,
                                                  embeddings=self.loaded_embeddings,
                                                  word_indices=self.word_indices)
        similarity = 1 - distance
        return similarity

    def build_statement_context_df(self, statement: str, context: str) -> pd.DataFrame:
        data = {'sentence0': context, 'sentence1': statement}
        data_processing.sentences_to_padded_index_sequences(self.word_indices, [[data]])
        data = pd.DataFrame([data])
        return data

    def get_confidences(self, unlabeled_confidences, labels: t.Dict) -> t.Dict:
        confidences = {
            labels[index]: unlabeled_confidences[index]
            for index in range(len(labels.keys()))
        }
        return confidences

    def predict_snli(self, data: pd.DataFrame):
        unlabeled = self.snli_classifier.continue_classify(data)
        return self.get_confidences(unlabeled, SNLI_INVERSE_MAP)

    def predict_terms(self, data: pd.DataFrame, terms: t.List[str]):
        density = term_scoring.get_term_density_in_text(data['sentence0'][0], terms)
        return density


@functools.lru_cache(maxsize=128*16)
def ensemble_is_proposition(context):
    """Gets if the given sentence is a proposition, and as such witha premise space tp calculate."""
    doc = textacy.make_spacy_doc(context, lang='en_core_web_sm')
    try:
        return is_proposition(doc)
    except AttributeError:
        init_weights()
        return is_proposition(doc)
