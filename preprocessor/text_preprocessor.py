import string
from typing import List

import util.text_util as text_util


class TextPreprocessResult:

    def __init__(self, bert_sentences: List[str]):
        self.bert_sentences = bert_sentences


class TextPreprocessor:

    def process(self, text: str) -> TextPreprocessResult:
        bert_sentences = self.__clean(text)
        return TextPreprocessResult(bert_sentences)

    def __clean(self, text: str) -> List[str]:
        # TODO: Conform to requirements
        sentences = []
        for sentence in text_util.tokenize_by_sentence(text):
            sentence = text_util.expand_contractions(sentence)
            sentence = text_util.replace_symbols(sentence)
            sentence = text_util.clean_accent(sentence)
            sentence = text_util.convert_nums_to_words(sentence)
            sent_words = []
            for word in text_util.tokenize_by_word(sentence):
                word = word.strip()
                if word not in set(string.punctuation):
                    sent_words.append(word)
            sentences.append(' '.join(sent_words))
        return sentences
