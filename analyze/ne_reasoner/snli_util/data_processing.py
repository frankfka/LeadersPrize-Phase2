import random
import json
import collections

from analyze.ne_reasoner import parameters, data_processing

FIXED_PARAMETERS = parameters.parameters


def load_nli_data(path, snli=False):
    """
    Load MultiNLI or SNLI data.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. 
    """
    data = []
    with open(path, 'r') as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"

            row = {'pairID': loaded_example['pairID'],
                   'sentence0': loaded_example['sentence1'],
                   'sentence1': loaded_example['sentence2'],
                   'label': loaded_example['label']}
            data.append(row)
        random.seed(1)
        random.shuffle(data)
    return data


def load_nli_data_genre(path, genre, snli=True):
    """
    Load a specific genre's examples from MultiNLI, or load SNLI data and assign a "snli" genre to the examples.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. If set to true, it will overwrite the genre label for MultiNLI data.
    """
    data = []
    j = 0
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            if loaded_example["genre"] == genre:
                data.append(loaded_example)
        random.seed(1)
        random.shuffle(data)
    return data


def build_dictionary(training_datasets):
    """
    Extract vocabulary and build dictionary.
    """
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            word_counter.update(data_processing.tokenize(example['sentence1_binary_parse']))
            word_counter.update(data_processing.tokenize(example['sentence2_binary_parse']))

    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    vocabulary = [data_processing.PADDING, data_processing.UNKNOWN] + vocabulary

    word_indices = dict(zip(vocabulary, range(len(vocabulary))))

    return word_indices


