import nltk
import numpy as np

PADDING = "<PAD>"
UNKNOWN = "<UNK>"


def tokenize(text: str):
    text = text.lower()
    return nltk.tokenize.word_tokenize(text)


def sentences_to_padded_index_sequences(word_indices, params, datasets):
    """
    Annotate datasets with feature vectors. Adding right-sided padding.
    """
    # print('start sentences_to_padded_index_sequences')
    for _, dataset in enumerate(datasets):
        for example in dataset:
            for sentence in ['sentence0', 'sentence1']:
                example[sentence + '_index_sequence'] = np.zeros((params["seq_length"]), dtype=np.int32)

                token_sequence = tokenize(example[sentence])
                for i in range(params["seq_length"]):
                    if i >= len(token_sequence):
                        index = word_indices[PADDING]
                    else:
                        try:
                            index = word_indices[token_sequence[i]]
                        except KeyError:
                            index = word_indices[UNKNOWN]

                        # if token_sequence[i] in word_indices:
                        #     index = word_indices[token_sequence[i]]
                        # else:
                        #     index = word_indices[UNKNOWN]
                    example[sentence + '_index_sequence'][i] = index
    # print('end sentences_to_padded_index_sequences')


def loadEmbedding_rand(path, params, word_indices):
    """
    Load GloVe embeddings. Doing a random normal initialization for OOV words.
    """
    np.random.seed(1)

    n = len(word_indices)
    m = params["word_embedding_dim"]
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
