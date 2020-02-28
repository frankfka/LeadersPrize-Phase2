from enum import Enum
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification

from reasoner.transformers.transformers_util import tokenize_for_transformer, logits_to_probabilities


class TransformersInputItem:

    def __init__(self, uuid, text_a, text_b, label=None):
        self.uuid = uuid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class ConfigKeys(Enum):
    TOK_PATH = "tokenizer_path"
    CONFIG_PATH = "config_path"
    MODEL_PATH = "model_path"
    NUM_LABELS = "num_labels"
    BATCH_SIZE = "batch_size"
    MAX_SEQ_LEN = "max_seq_len"


class RobertaSequenceClassifier:

    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = RobertaTokenizer.from_pretrained(config[ConfigKeys.TOK_PATH])
        config = RobertaConfig.from_pretrained(config[ConfigKeys.CONFIG_PATH], num_labels=config[ConfigKeys.NUM_LABELS])
        self.model = RobertaForSequenceClassification.from_pretrained(config[ConfigKeys.MODEL_PATH], config=config)
        self.model.eval()
        self.device = "cpu"
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"

    def predict(self, input_items: List[TransformersInputItem], debug=False):
        """
        Get predictions for the inputs. Each prediction is a numpy array of the probabilities of each class
        """
        # Get tokenized inputs
        token_ids, masks, token_type_ids = tokenize_for_transformer(input_items,
                                                                    self.tokenizer,
                                                                    self.config[ConfigKeys.MAX_SEQ_LEN],
                                                                    debug=debug)
        # Get PyTorch data loader
        dataset = TensorDataset(token_ids, masks, token_type_ids)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.config[ConfigKeys.BATCH_SIZE])

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
                probabilities = logits_to_probabilities(prediction).numpy()
                predictions.append(probabilities)

        return np.array(predictions)
