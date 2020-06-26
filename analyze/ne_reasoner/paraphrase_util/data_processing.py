import csv
import pandas as pd
import numpy as np
import random
import collections

import typing

from analyze.ne_reasoner import parameters, data_processing

FIXED_PARAMETERS = parameters.parameters

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": 0
}

PARAPHRASE_INVERSE_MAP = {
    0: "not_paraphrase",
    1: "paraphrase",
}


def split_paraphrase_data(data: pd.DataFrame, train_amount, dev_amount
                          ) -> typing.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end_index = int(train_amount * len(data))
    dev_end_index = int((train_amount + dev_amount) * len(data))
    np.random.seed(1)
    train_test_validate = np.split(data.sample(frac=1), [train_end_index, dev_end_index])
    return train_test_validate


def load_paraphrase_data(path) -> pd.DataFrame:
    """
    Load MSR Paraphrase data.
    """

    rows = []
    with open(path, 'r', encoding='UTF-8', errors='ignore') as f:
        reader = csv.reader(f, delimiter='\t', quotechar='|')
        next(reader)  # skip header
        for line in reader:
            # noinspection PyDictCreation
            row = {}
            row['is_paraphrase'], row['id0'], row['id1'], row['sentence0'], row['sentence1'] = line
            rows.append(row)
        random.seed(1)
        random.shuffle(rows)
    data = pd.DataFrame(rows)
    return data


def build_dictionary(training_datasets):
    """
    Extract vocabulary and build dictionary.
    """
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            sentence_0_tokens = data_processing.tokenize(example['sentence0'])
            sentence_1_tokens = data_processing.tokenize(example['sentence1'])

            word_counter.update(sentence_0_tokens)
            word_counter.update(sentence_1_tokens)

    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    vocabulary = [data_processing.PADDING, data_processing.UNKNOWN] + vocabulary

    word_indices = dict(zip(vocabulary, range(len(vocabulary))))

    return word_indices


def sentences_to_padded_index_sequences(word_indices, datasets: typing.List[pd.DataFrame]):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """

    def get_token_sequence(text: str):
        tokens = data_processing.tokenize(text)
        indices = np.zeros((FIXED_PARAMETERS["seq_length"]), dtype=np.int32)

        for i in range(FIXED_PARAMETERS["seq_length"]):
            if i >= len(tokens):
                index = word_indices[data_processing.PADDING]
            elif tokens[i] in word_indices:
                index = word_indices[tokens[i]]
            else:
                index = word_indices[data_processing.UNKNOWN]
            indices[i] = index
        return indices

    for i, dataset in enumerate(datasets):
        # for example in dataset.iterrows():
        for sentence in ['sentence0', 'sentence1']:
            dataset[f'{sentence}_index_sequence'] = \
                np.vectorize(get_token_sequence, otypes=[object])(dataset[sentence])
