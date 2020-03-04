from typing import List
import torch
from reasoner.transformers.models import TransformersInputItem


# Preprocesses text into inputs expected for the model (token_ids, attention_masks, token_type_ids)
def tokenize_for_transformer(input_items: List[TransformersInputItem],
                             tokenizer,
                             max_seq_len: int,
                             debug=False):
    # XLNet: A [SEP] B [SEP][CLS]
    # BERT: [CLS] A [SEP] B [SEP]
    # Roberta:  <s> A </s></s> B </s>
    # ALBERT: [CLS] A [SEP] B [SEP]

    all_token_ids = []
    all_attention_masks = []
    all_token_types = []

    for item in input_items:

        tokenized_input = tokenizer.encode_plus(
            text=item.text_a,
            text_pair=item.text_b,
            add_special_tokens=True,
            max_length=max_seq_len,
            stride=0,
            truncation_strategy="only_second",
            pad_to_max_length=True,
            return_tensors=None,  # Can set this to "pt to get a torch tensor back
            return_token_type_ids=True,
            return_attention_mask=True,
            return_overflowing_tokens=False,
            return_special_tokens_mask=False,
            return_offsets_mapping=False
        )
        input_ids, type_ids, masks = tokenized_input["input_ids"], tokenized_input["token_type_ids"], tokenized_input[
            "attention_mask"]

        assert len(input_ids) == max_seq_len
        assert len(type_ids) == max_seq_len
        assert len(masks) == max_seq_len

        # Print an example for debugging
        if debug:
            print(f"Token ID's: {input_ids}")
            print(f"Mask: {masks}")
            print(f"Token Type ID's: {type_ids}")
            print("\n")
            debug = False

        all_token_ids.append(input_ids)
        all_attention_masks.append(masks)
        all_token_types.append(type_ids)

    all_token_ids = torch.tensor(all_token_ids, dtype=torch.long)
    all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)
    all_token_types = torch.tensor(all_token_types, dtype=torch.long)

    return all_token_ids, all_attention_masks, all_token_types
