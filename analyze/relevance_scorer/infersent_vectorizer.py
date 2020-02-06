import time
from typing import List

from analyze.relevance_scorer.infersent.models import InferSent
import torch


class InfersentVectorizer(object):
    """
    An Infersent V2 vectorizer that vectorizes sentences using FastText embeddings
    """

    def __init__(self, infersent_model_path, fasttext_model_path, vocab_count=500000, use_cuda=False):
        start_time = time.time()
        model_params = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}
        # Load Infersent model
        print("Loading Infersent model")
        model = InferSent(model_params)
        model.load_state_dict(torch.load(infersent_model_path))
        model = model.cuda() if use_cuda else model
        # Load word vectors
        model.set_w2v_path(fasttext_model_path)
        # Load embeddings of K most frequent words
        print("Building Infersent vocab")
        model.build_vocab_k_words(K=vocab_count)
        self.model = model
        print(f"Infersent vectorizer loaded in {time.time() - start_time}s")

    def get_sentence_vectors(self, sentences: List[str]):
        return self.model.encode(sentences, tokenize=False, verbose=False)
