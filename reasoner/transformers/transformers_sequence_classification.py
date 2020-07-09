from enum import Enum
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizerFast

from reasoner.models import TransformersInputItem
from reasoner.transformers.transformers_util import tokenize_for_transformer


class TransformersConfigKeys(Enum):
    TOK_PATH = "tokenizer_path"
    CONFIG_PATH = "config_path"
    MODEL_PATH = "model_path"
    NUM_LABELS = "num_labels"
    BATCH_SIZE = "batch_size"
    MAX_SEQ_LEN = "max_seq_len"


class RobertaSequenceClassifier:

    def __init__(self, config: Dict):
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.config = config
        self.tokenizer = RobertaTokenizerFast.from_pretrained(config[TransformersConfigKeys.TOK_PATH],
                                                              add_prefix_space=True)
        model_config = RobertaConfig.from_pretrained(config[TransformersConfigKeys.CONFIG_PATH],
                                                     num_labels=config[TransformersConfigKeys.NUM_LABELS])
        self.model = RobertaForSequenceClassification.from_pretrained(config[TransformersConfigKeys.MODEL_PATH],
                                                                      config=model_config)
        self.model.eval()
        self.device = "cpu"
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"

    def predict(self, text_a_arr: List[str], text_b_arr: List[str]):
        """
        Get predictions for the inputs. Each prediction is a numpy array of the probabilities of each class
        """
        # Get tokenized inputs
        token_ids, masks, token_type_ids = tokenize_for_transformer(text_a_arr, text_b_arr, self.tokenizer)
        # Get PyTorch data loader
        dataset = TensorDataset(token_ids, masks, token_type_ids)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.config[TransformersConfigKeys.BATCH_SIZE])

        # Run to get predictions
        predictions = []
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2]
            }

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs[0]

            logits = logits.detach().cpu()
            for prediction in logits:
                predictions.append(prediction.numpy())

        return np.array(predictions)
