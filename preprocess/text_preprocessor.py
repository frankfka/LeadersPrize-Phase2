from typing import List

import preprocess.text_util as text_util


class TextPreprocessResult:

    def __init__(self, sentences: List[str]):
        self.sentences = sentences


class TextPreprocessor:

    def process_sentences(self, sentences: List[str]) -> List[str]:
        return [self.process_one_sentence(sent) for sent in sentences]

    def process_one_sentence(self, text: str) -> str:
        cleaned_sentence = text_util.expand_contractions(text)
        cleaned_sentence = text_util.replace_symbols(cleaned_sentence)
        cleaned_sentence = text_util.clean_accent(cleaned_sentence)
        cleaned_sentence = text_util.convert_num_to_words_v2(cleaned_sentence)
        return cleaned_sentence.strip()
