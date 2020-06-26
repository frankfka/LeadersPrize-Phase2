import functools

import pandas as pd
import typing
import typing as t
import textacy

from analyze.ne_reasoner import vocab, data_processing, term_scoring, similarity_mod, predict_snli, snli_util, \
    logger_mod, propositions


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

    def __init__(self):
        self.data_root_path = '../data'
        self.word_indices = vocab.load_dictionary()
        self.embedding_path = f'{self.data_root_path}/glove.6B.50d.txt'
        self.loaded_embeddings = data_processing.loadEmbedding_rand(self.embedding_path, self.word_indices)

        self.snli_classifier = self.init_snli()

        # todo move to separate classes

    def predict_statement_in_contexts(self, statement: str, contexts: typing.List[str]):
        lines = []
        predictions = []
        for i, context in enumerate(contexts):
            if self.is_relevant(statement, context) and is_proposition(context):
                prediction = self.predict_statement_in_context(statement, context)
                lines.append(i)
                predictions.append(prediction)
        return lines, predictions

    def predict_statement_in_context(self, statement: str, context: str):
        predictions = {}

        data = self.build_statement_context_df(statement, context)
        predictions.update(self.predict_snli(data))

        predictions['paraphrase'] = float(self.is_paraphrase(statement, context))

        terms = {column_name: term_scoring.get_terms(column_name)
                 for column_name in self.term_column_names}
        term_predictions = {column_name: self.predict_terms(data, terms[column_name])
                            for column_name in self.term_column_names}
        unskewed = {column_name: term_scoring.unskew_term_predictions(column_name, term_predictions[column_name])
                    for column_name in self.term_column_names}
        predictions.update(unskewed)

        return predictions

    def is_relevant(self, statement, context) -> bool:
        return self._check_similarity(statement, context, 0.65)

    def is_paraphrase(self, statement, context) -> bool:
        return self._check_similarity(statement, context, 0.80)

    def _check_similarity(self, statement, context, similarity_threshold) -> bool:
        distance = similarity_mod.cosine_distance(statement,
                                                  context,
                                                  embeddings=self.loaded_embeddings,
                                                  word_indices=self.word_indices)
        similarity = 1 - distance
        is_similar = similarity >= similarity_threshold
        return is_similar

    def build_statement_context_df(self, statement: str, context: str) -> pd.DataFrame:
        data = {'sentence0': context, 'sentence1': statement}
        data_processing.sentences_to_padded_index_sequences(self.word_indices, [[data]])
        data = pd.DataFrame([data])
        return data

    def init_snli(self):
        snli_classifier = predict_snli.SnliClassifier(
            loaded_embeddings=self.loaded_embeddings,
            processing=snli_util.data_processing,
            logger=logger_mod.Logger(),
            modname='ensemble_snli',
            emb_train=True)
        snli_classifier.restore()
        return snli_classifier

    def get_confidences(self, unlabeled_confidences, labels: t.Dict) -> t.Dict:
        confidences = {
            labels[index]: unlabeled_confidences[index]
            for index in range(len(labels.keys()))
        }
        return confidences

    def predict_snli(self, data: pd.DataFrame):
        unlabeled = self.snli_classifier.continue_classify(data)
        return self.get_confidences(unlabeled, snli_util.data_processing.SNLI_INVERSE_MAP)

    def predict_terms(self, data: pd.DataFrame, terms: t.List[str]):
        density = term_scoring.get_term_density_in_text(data['sentence0'][0], terms)
        return density


@functools.lru_cache(maxsize=128*16)
def is_proposition(context):
    """Gets if the given sentence is a proposition, and as such witha premise space tp calculate."""
    doc = textacy.make_spacy_doc(context, lang='en_core_web_sm')
    try:
        return propositions.main.is_proposition(doc)
    except AttributeError:
        propositions.main.init_weights()
        return propositions.main.is_proposition(doc)
