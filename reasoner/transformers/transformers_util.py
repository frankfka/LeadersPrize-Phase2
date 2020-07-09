from typing import List

from transformers import RobertaTokenizerFast


# Preprocesses text into inputs expected for the model (token_ids, attention_masks, token_type_ids)
def tokenize_for_transformer(text_a_arr: List[str], text_b_arr: List[str], tokenizer: RobertaTokenizerFast):
    tokenized_result = tokenizer(
        text_a_arr,
        text_b_arr,
        padding='max_length',
        truncation='only_second',
        return_tensors="pt",
        return_token_type_ids=True
    )

    return tokenized_result['input_ids'], tokenized_result['attention_mask'], tokenized_result['token_type_ids']
