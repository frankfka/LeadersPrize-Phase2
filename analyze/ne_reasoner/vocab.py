"""
Extract vocabulary and build dictionary.
"""
import collections
import pickle
import glob

import nltk
import typing

PADDING = "<PAD>"
UNKNOWN = "<UNK>"

save_path = "../data/ensemble_vocab.p"

def tokenize(text: str):
    text = text.lower()
    return nltk.tokenize.word_tokenize(text)


def build_dictionary():
    word_counter = collections.Counter()
    return word_counter


def update_dictionary(text, dictionary):
    tokenized = tokenize(text)
    dictionary.update(tokenized)

    return dictionary


def save_dictionary(dictionary):
    vocabulary = set([word for word in dictionary])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING, UNKNOWN] + vocabulary
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))
    pickle.dump(word_indices, open(save_path, "wb+"))
    return word_indices


def build_from_all_training_data_in_sources(sources: typing.List[str]):
    dictionary = build_dictionary()
    for rel_path in sources:
        matcher = f'../data/{rel_path}/**/*.*'
        paths = glob.glob(matcher, recursive=True)
        for path in paths:
            with open(path, 'r', errors='ignore') as f:
                lines = f.readlines()
                for line in lines:
                    update_dictionary(line, dictionary)

    save_dictionary(dictionary)
    return dictionary

def load_dictionary():
    with open(save_path, 'rb') as f:
        dictionary = pickle.load(f)
        return dictionary


if __name__ == '__main__':


    dictionary = load_dictionary()

    print('debug')
