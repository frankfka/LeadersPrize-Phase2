import nltk
import numpy as np

from analyze.ne_reasoner import parameters

FIXED_PARAMETERS = parameters.parameters

PADDING = "<PAD>"
UNKNOWN = "<UNK>"


def tokenize(text: str):
    text = text.lower()
    return nltk.tokenize.word_tokenize(text)


def sentences_to_padded_index_sequences(word_indices, datasets):
    """
    Annotate datasets with feature vectors. Adding right-sided padding.
    """
    for _, dataset in enumerate(datasets):
        for example in dataset:
            for sentence in ['sentence0', 'sentence1']:
                _sentence_to_padded_index_sequence(sentence, example, word_indices)


def _sentence_to_padded_index_sequence(sentence, example, word_indices):
    example[sentence + '_index_sequence'] = np.zeros((FIXED_PARAMETERS["seq_length"]), dtype=np.int32)
    token_sequence = tokenize(example[sentence])
    for i in range(FIXED_PARAMETERS["seq_length"]):
        if i >= len(token_sequence):
            index = word_indices[PADDING]
        else:
            try:
                index = word_indices[token_sequence[i]]
            except KeyError:
                index = word_indices[UNKNOWN]

        example[sentence + '_index_sequence'][i] = index


def loadEmbedding_zeros(path, word_indices):
    """
    Load GloVe embeddings. Initializing OOV words to vector of zeros.
    """
    emb = np.zeros((len(word_indices), FIXED_PARAMETERS["word_embedding_dim"]), dtype='float32')

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if FIXED_PARAMETERS["embeddings_to_load"] is not None and i >= FIXED_PARAMETERS["embeddings_to_load"]:
                break

            s = line.split()
            if s[0] in word_indices:
                emb[word_indices[s[0]], :] = np.asarray(s[1:])

    return emb


def loadEmbedding_rand(path, word_indices):
    """
    Load GloVe embeddings. Doing a random normal initialization for OOV words.
    """
    np.random.seed(1)

    n = len(word_indices)
    m = FIXED_PARAMETERS["word_embedding_dim"]
    emb = np.empty((n, m), dtype=np.float32)

    emb[:, :] = np.random.normal(size=(n, m))

    # Explicitly assign embedding of <PAD> to be zeros.
    emb[0:2, :] = np.zeros((1, m), dtype="float32")

    with open(path, 'r', encoding="utf8") as f:
        for i, line in enumerate(f):
            s = line.split()
            if s[0] in word_indices:
                emb[word_indices[s[0]], :] = np.asarray(s[1:])

    return emb
