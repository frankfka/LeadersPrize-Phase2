import re
import sys
import math
import concurrent.futures
import typing
from typing import List
import numpy as np
from scipy import spatial
from scipy.sparse import dok_matrix
from scipy.sparse import dok
import scipy.sparse.linalg as ssl
from tqdm import tqdm
from numba import jit


# TODO: Put these as part of preprocessing pipeline
def __clean_line__(line: str) -> str:
    """
    This module is responsible for converting a document into a cleaned version using nltk
    1. Tokenize using whitespace splitting
    2. Remove punctuation
    """
    tokens = re.sub('[^a-z0-9 ]+', ' ', line.lower()).split(' ')
    stemmed_tokens = [w for w in tokens if len(w) > 2]
    return ' '.join(stemmed_tokens)


def __clean_text__(lines: np.ndarray, tokenizer=None, stemmer=None) -> np.ndarray:
    """
    This module is responsible for converting a document into a cleaned version using nltk.
    Same as above except handles list of lines
    """
    clean_lines = np.empty(len(lines), dtype=object)
    for i, line in enumerate(lines):
        clean_lines[i] = __clean_line__(line)
    return clean_lines


class LSADocumentRelevanceAnalyzer:
    """
    For now, this
    """

    def __init__(self):
        pass  # create any dependencies

    def analyze(self, claim: str, articles: typing.List[str], decomposition_k=20) -> typing.List[float]:
        """Determine the relevance of the claim to each article, and return those relevances in order."""
        # k>20 ensures better semantic computation of local and global sem.spaces

        if len(articles) < decomposition_k:
            decomposition_k = min(decomposition_k, len(articles) - 1)

        workers = 1
        documents = __read_raw_docs__([claim] + articles, -1, workers)

        words = __get_unique_words__(documents, workers)
        doc_matrix, documents = __get_sparse_matrix__(documents, words, workers)

        try:
            u, s, vt = __decomposition__(doc_matrix, decomposition_k)
        except ValueError:
            return []

        relevances = __doc_comparisons__(u, s, vt, documents, False, doc_matrix.todense().T)
        relevances = relevances[1:]  # Remove claim from results.
        return relevances


# @j
def __decomposition__(docmatrix, k):
    k = min(k, min(docmatrix.shape) - 1)
    u, s, vt = ssl.svds(docmatrix.T, k=k)

    return u, s, vt


def __doc_comparisons__(u, s, vt, documents, output, docmatrix):
    d = np.diag(s)
    num_docs = vt.shape[1]
    doc_mat = d @ vt

    error = False

    try:
        query_str = 'Enter the document number you wish to query (between 0 and {} inclusive): '
        # index = int(input(query_str.format(len(documents) - 1)))
        index = 0
        if index >= len(documents) or index < 0:
            error = True
        # print('You Queried: {}'.format(documents[index]))
    except ValueError:
        print("Insert Valid Number")
        error = True

    distance_func = lambda a, b: 1 - spatial.distance.cosine(a, b)
    # distance_func = lambda a, b: sct.spearmanr(a, b)[0]

    if not error:
        q = doc_mat[:, index]

        rank = np.zeros(num_docs)

        relevances = []
        for i in range(num_docs):
            relevance = distance_func(doc_mat[:, i], q)
            rank[i] = relevance
            relevances.append(relevance)
        return relevances


def __read_raw_docs__(lines: List[str], size: int, workers: int) -> np.ndarray:
    if size == -1:
        size = len(lines)
    lines = lines[:size]
    documents = np.empty(size, dtype=object)
    memory_impact = sum([sys.getsizeof(s) for s in lines])
    # j
    # rec
    if memory_impact < 50000000:
        offset = 0
        linebins = np.array_split(lines, workers)  # this is the offending large memory line
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(__clean_text__, linebins[i]): i
                       for i in range(workers)}
            for future in tqdm(concurrent.futures.as_completed(futures),
                               desc='Tokenizing Documents', total=workers,
                               leave=True, disable=True):
                index = futures[future]
                for i, line in enumerate(future.result()):
                    documents[offset + i] = line
                offset += len(future.result())
    else:
        print('Use Large Memory Algorithm')
        offset = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(__clean_line__, lines[i]): i
                       for i in range(size)}
            for future in tqdm(concurrent.futures.as_completed(futures),
                               desc='Tokenizing Documents', total=size,
                               leave=True, disable=True):
                documents[offset] = future.result()
                offset += 1
    return documents


# @enforce.runtime_validation
def __get_unique_words__(documents: np.ndarray, workers: int) -> dict:
    """
    Parallelize Unique Word Calculation

    :documents: list of document strings
    :workers: number of workers

    :return: dictionary of ngramfrequencies
    """
    data_bins = np.array_split(documents, workers)
    wordlist = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(__unique_words__, data_bins[i]): i for i in
                   range(workers)}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           desc='Determining Unique Words', leave=True,
                           total=workers, disable=True):
            for word, stats in future.result().items():
                try:
                    wordlist[word]['freq'] += stats['freq']
                    wordlist[word]['doccount'] += stats['doccount']
                except KeyError:
                    wordlist[word] = {'freq': stats['freq'],
                                      'doccount': stats['doccount']}
    return wordlist


def __unique_words__(data: np.ndarray) -> dict:
    """
    :data: list of document strings
    :return: dictionary of word frequencies
    """
    words = {}
    for doc in data:
        docwords_list = [w for w in doc.split(' ') if w != '']
        docwords_set = set(docwords_list)
        for word in docwords_list:
            try:
                words[word]['freq'] += 1
            except KeyError:
                words[word] = {'freq': 1, 'doccount': 0}
        for word in docwords_set:
            words[word]['doccount'] += 1
    return words


@jit
def __weight__(total_doc_count: int, doccount: int, wordfreq: int) -> float:
    """
    Weighting function for Document Term Matrix.
    tf-idf
    """
    return (1 + math.log(wordfreq)) * (math.log(total_doc_count / doccount))


# @enforce.runtime_validation
def __get_sparse_matrix__(documents: np.ndarray, words: dict, workers: int,
                          weighting: typing.Any = __weight__) -> typing.Tuple[
    dok.dok_matrix, np.ndarray]:
    """
    Parallelize Sparse Matrix Calculation
    :documents: list of document strings
    :words: dictionary of word frequencies
    :workers: number of workers
    :return: Sparse document term matrix
    """
    m = len(documents)
    n = len(words.keys())
    # Make sure we don't have more bins than workers
    workers = m if m < workers else workers
    data_bins = np.array_split(documents, workers)
    docmatrix = dok_matrix((m, n), dtype=float)
    new_docs = np.empty(len(documents), dtype=object)
    offsets = [len(data_bin) for data_bin in data_bins]
    coffset = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(__parse_docs__,
                                   data_bins[i],
                                   words,
                                   len(documents),
                                   weighting): i
                   for i in range(workers)}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           desc='Parsing Documents and Combining Arrays',
                           leave=True, total=workers, disable=True):
            binnum = futures[future]
            # Because order is not preserved, we need to make sure we add
            # the documents back in the correct order.
            for i, doc in enumerate(data_bins[binnum]):
                new_docs[coffset + i] = doc
            # THIS IS THE BOTTLENECK
            for key, value in future.result().items():
                docmatrix[key[0] + coffset, key[1]] = value
            coffset += offsets[binnum]
    return docmatrix, new_docs


def __parse_docs__(data: np.ndarray, words: dict, doc_count: int,
                   weight_func: typing.Any) -> dict:
    """
    Parallelize Sparse Matrix Calculation
    :data: list of document strings
    :words: dictionary of word frequencies
    :total_doc_count: total number of documents (for tf-idf)
    :weight_func: weighting function for code
    :return: sparse array with weighted values
    """
    m = len(data)
    n = len(words.keys())
    docmatrix = {}
    wordref = {w: i for i, w in enumerate(sorted(words.keys()))}
    for i, doc in enumerate(data):
        for word in list(set(doc.split(' '))):
            if word != '':
                docmatrix[(i, wordref[word])] = weight_func(doc_count,
                                                            words[word][
                                                                'doccount'],
                                                            words[word]['freq'])
    return docmatrix
